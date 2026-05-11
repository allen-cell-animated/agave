"""Tests for AGAVE discovery / auto-launch ordering logic."""

import subprocess
from unittest import mock

import pytest

from agave_pyclient import find_agave


# ---------------------------------------------------------------------------
# port_from_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("ws://localhost:1235/", 1235),
        ("ws://localhost:9000", 9000),
        ("ws://example.com:42/path", 42),
        ("ws://localhost/", 1235),  # no port -> default
        ("not a url", 1235),
    ],
)
def test_port_from_url(url, expected):
    assert find_agave.port_from_url(url) == expected


def test_port_from_url_custom_default():
    assert find_agave.port_from_url("ws://localhost/", default=9999) == 9999


# ---------------------------------------------------------------------------
# guess_agave_path: PATH is checked first
# ---------------------------------------------------------------------------


def test_guess_agave_path_prefers_PATH(monkeypatch):
    """If `agave` is on PATH, that wins over any standard install location."""
    monkeypatch.setattr(
        find_agave.shutil, "which", lambda name: "/usr/local/bin/agave"
    )
    # Make sure platform-specific fallbacks would also "succeed" — PATH still wins.
    monkeypatch.setattr(find_agave.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(find_agave.sys, "platform", "linux")

    assert find_agave.guess_agave_path() == "/usr/local/bin/agave"


def test_guess_agave_path_falls_back_to_standard_locations(monkeypatch):
    """When PATH lookup fails, scan the per-platform candidate list."""
    monkeypatch.setattr(find_agave.shutil, "which", lambda name: None)
    monkeypatch.setattr(find_agave.sys, "platform", "linux")

    # Only /opt/agave/agave "exists".
    existing = {"/opt/agave/agave"}
    monkeypatch.setattr(find_agave.os.path, "isfile", lambda p: p in existing)

    assert find_agave.guess_agave_path() == "/opt/agave/agave"


def test_guess_agave_path_returns_none_when_nothing_found(monkeypatch):
    monkeypatch.setattr(find_agave.shutil, "which", lambda name: None)
    monkeypatch.setattr(find_agave.sys, "platform", "linux")
    monkeypatch.setattr(find_agave.os.path, "isfile", lambda p: False)

    assert find_agave.guess_agave_path() is None


def test_guess_agave_path_unknown_platform_returns_none(monkeypatch):
    monkeypatch.setattr(find_agave.shutil, "which", lambda name: None)
    monkeypatch.setattr(find_agave.sys, "platform", "plan9")

    assert find_agave.guess_agave_path() is None


# ---------------------------------------------------------------------------
# resolve_agave_path: explicit path beats guessing
# ---------------------------------------------------------------------------


def test_resolve_agave_path_uses_explicit_over_guess(monkeypatch):
    monkeypatch.setattr(find_agave.os.path, "isfile", lambda p: True)
    # guess_agave_path should NOT be consulted when an explicit path is given.
    sentinel = mock.Mock(side_effect=AssertionError("guess should not be called"))
    monkeypatch.setattr(find_agave, "guess_agave_path", sentinel)

    assert find_agave.resolve_agave_path("/explicit/agave") == "/explicit/agave"
    sentinel.assert_not_called()


def test_resolve_agave_path_uses_guess_when_none(monkeypatch):
    monkeypatch.setattr(find_agave, "guess_agave_path", lambda: "/guessed/agave")
    monkeypatch.setattr(find_agave.os.path, "isfile", lambda p: True)

    assert find_agave.resolve_agave_path(None) == "/guessed/agave"


def test_resolve_agave_path_raises_when_nothing_found(monkeypatch):
    monkeypatch.setattr(find_agave, "guess_agave_path", lambda: None)
    with pytest.raises(RuntimeError, match="no installed executable"):
        find_agave.resolve_agave_path(None)


def test_resolve_agave_path_raises_when_explicit_missing(monkeypatch):
    monkeypatch.setattr(find_agave.os.path, "isfile", lambda p: False)
    with pytest.raises(FileNotFoundError):
        find_agave.resolve_agave_path("/does/not/exist")


# ---------------------------------------------------------------------------
# launch_agave_process: subprocess invocation
# ---------------------------------------------------------------------------


def test_launch_agave_process_invokes_subprocess(monkeypatch):
    captured = {}

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return mock.Mock(spec=subprocess.Popen)

    monkeypatch.setattr(find_agave.subprocess, "Popen", fake_popen)

    find_agave.launch_agave_process("/path/to/agave", port=4242)

    assert captured["cmd"] == ["/path/to/agave", "--server", "--port=4242"]


def test_launch_agave_process_translates_oserror(monkeypatch):
    def fake_popen(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr(find_agave.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="Error launching AGAVE"):
        find_agave.launch_agave_process("/path/to/agave")


# ---------------------------------------------------------------------------
# wait_for_connection: retry / cleanup ordering
# ---------------------------------------------------------------------------


def _proc(poll_returns=None, returncode=0):
    """Build a fake Popen-like object whose `poll()` returns successive values."""
    p = mock.Mock(spec=subprocess.Popen)
    p.poll.side_effect = list(poll_returns or [None, None, None, None])
    p.returncode = returncode
    return p


def test_wait_for_connection_returns_first_success(monkeypatch):
    monkeypatch.setattr(find_agave.time, "sleep", lambda s: None)
    proc = _proc()
    connect = mock.Mock(return_value="ok")

    assert find_agave.wait_for_connection(connect, proc, retries=5) == "ok"
    connect.assert_called_once()
    proc.terminate.assert_not_called()


def test_wait_for_connection_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(find_agave.time, "sleep", lambda s: None)
    proc = _proc()
    connect = mock.Mock(side_effect=[ConnectionError("x"), ConnectionError("x"), "ok"])

    assert find_agave.wait_for_connection(connect, proc, retries=5) == "ok"
    assert connect.call_count == 3
    proc.terminate.assert_not_called()


def test_wait_for_connection_terminates_on_exhaustion(monkeypatch):
    monkeypatch.setattr(find_agave.time, "sleep", lambda s: None)
    proc = _proc()
    connect = mock.Mock(side_effect=ConnectionError("nope"))

    with pytest.raises(RuntimeError, match="Could not connect"):
        find_agave.wait_for_connection(connect, proc, retries=3)

    assert connect.call_count == 3
    proc.terminate.assert_called_once()
    proc.wait.assert_called_once()


def test_wait_for_connection_detects_dead_process(monkeypatch):
    monkeypatch.setattr(find_agave.time, "sleep", lambda s: None)
    # poll() returns a non-None exit code immediately -> process died.
    proc = _proc(poll_returns=[7], returncode=7)
    connect = mock.Mock(return_value="should-not-be-called")

    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        find_agave.wait_for_connection(connect, proc, retries=5)

    connect.assert_not_called()


# ---------------------------------------------------------------------------
# AgaveRenderer.__init__ ordering
#
# The constructor should:
#   1. Try to connect to a running server.
#   2. If that fails AND auto_launch=True, resolve a path and launch.
#   3. After launching, retry the connection via wait_for_connection.
#   4. If auto_launch=False, never resolve/launch and re-raise.
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_renderer(monkeypatch):
    """Stub out everything AgaveRenderer.__init__ touches except its own logic."""
    from agave_pyclient import agave as agave_mod

    calls = []

    def record(name, *, return_value=None, side_effect=None):
        def fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            if side_effect is not None:
                if isinstance(side_effect, Exception):
                    raise side_effect
                return side_effect(*args, **kwargs)
            return return_value

        return fn

    fake_proc = mock.Mock(spec=subprocess.Popen)
    fake_proc.poll.return_value = None

    monkeypatch.setattr(
        agave_mod, "resolve_agave_path",
        record("resolve_agave_path", return_value="/resolved/agave"),
    )
    monkeypatch.setattr(
        agave_mod, "launch_agave_process",
        record("launch_agave_process", return_value=fake_proc),
    )
    monkeypatch.setattr(
        agave_mod, "wait_for_connection",
        record("wait_for_connection", return_value=None),
    )
    monkeypatch.setattr(agave_mod, "port_from_url", record("port_from_url", return_value=1235))

    return agave_mod, calls, fake_proc


def _make_renderer_skipping_init(agave_mod):
    """Bypass __init__ and call it manually in each test for fine control."""
    return agave_mod.AgaveRenderer.__new__(agave_mod.AgaveRenderer)


def test_init_connects_without_launching_when_server_running(patched_renderer, monkeypatch):
    agave_mod, calls, _ = patched_renderer
    # First _connect call succeeds.
    monkeypatch.setattr(
        agave_mod.AgaveRenderer, "_connect", lambda self: None
    )

    r = _make_renderer_skipping_init(agave_mod)
    agave_mod.AgaveRenderer.__init__(r)

    names = [c[0] for c in calls]
    # No path resolution, no launch, no waiting — we connected on the first try.
    assert "resolve_agave_path" not in names
    assert "launch_agave_process" not in names
    assert "wait_for_connection" not in names
    assert r.agave_process is None


def test_init_falls_through_to_launch_in_correct_order(patched_renderer, monkeypatch):
    agave_mod, calls, fake_proc = patched_renderer

    # First _connect call fails, all subsequent steps are stubbed.
    connect_calls = []

    def failing_connect(self):
        connect_calls.append(1)
        raise ConnectionRefusedError("no server")

    monkeypatch.setattr(agave_mod.AgaveRenderer, "_connect", failing_connect)

    r = _make_renderer_skipping_init(agave_mod)
    agave_mod.AgaveRenderer.__init__(r, agave_path="/explicit/agave")

    # Order matters: resolve -> port_from_url -> launch -> wait_for_connection.
    names = [c[0] for c in calls]
    assert names == [
        "resolve_agave_path",
        "port_from_url",
        "launch_agave_process",
        "wait_for_connection",
    ]
    # The explicit path was forwarded.
    assert calls[0][1] == ("/explicit/agave",)
    # The process from launch_agave_process was retained on self.
    assert r.agave_process is fake_proc


def test_init_auto_launch_false_reraises_without_launching(patched_renderer, monkeypatch):
    agave_mod, calls, _ = patched_renderer

    def failing_connect(self):
        raise ConnectionRefusedError("no server")

    monkeypatch.setattr(agave_mod.AgaveRenderer, "_connect", failing_connect)

    r = _make_renderer_skipping_init(agave_mod)
    with pytest.raises(ConnectionRefusedError):
        agave_mod.AgaveRenderer.__init__(r, auto_launch=False)

    # None of the launch helpers should have been touched.
    names = [c[0] for c in calls]
    assert "resolve_agave_path" not in names
    assert "launch_agave_process" not in names
    assert "wait_for_connection" not in names


def test_init_clears_process_when_wait_for_connection_fails(patched_renderer, monkeypatch):
    agave_mod, calls, fake_proc = patched_renderer

    monkeypatch.setattr(
        agave_mod.AgaveRenderer,
        "_connect",
        lambda self: (_ for _ in ()).throw(ConnectionRefusedError("no server")),
    )

    # Make wait_for_connection raise; AgaveRenderer should null out agave_process
    # (wait_for_connection is responsible for terminate/wait).
    def boom(*args, **kwargs):
        calls.append(("wait_for_connection", args, kwargs))
        raise RuntimeError("Could not connect to AGAVE after 1 attempts: x")

    monkeypatch.setattr(agave_mod, "wait_for_connection", boom)

    r = _make_renderer_skipping_init(agave_mod)
    with pytest.raises(RuntimeError, match="Could not connect"):
        agave_mod.AgaveRenderer.__init__(r)

    assert r.agave_process is None
