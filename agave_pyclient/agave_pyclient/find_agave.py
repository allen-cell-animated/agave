"""
Helpers for locating an installed AGAVE executable and launching it as a
local server process.

This module is intentionally free of any websocket / rendering dependencies
so it can be imported cheaply (and reused by other tooling).
"""

import os
import re
import shutil
import subprocess
import sys
import time
from typing import List, Optional


def find_matching_subdirectories(root_dir: str, regex: str) -> List[str]:
    """
    Find subdirectories within ``root_dir`` whose names match ``regex``.

    Returns a list of full paths. Returns an empty list if none match.
    """
    matching_dirs: List[str] = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and re.search(regex, item):
            matching_dirs.append(item_path)
    return matching_dirs


def port_from_url(url: str, default: int = 1235) -> int:
    """Extract a port number from a websocket URL, falling back to ``default``."""
    m = re.search(r":(\d+)(?:/|$)", url)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return default


def guess_agave_path() -> Optional[str]:
    """
    Try to locate an installed AGAVE executable on the current platform.

    Search order:
      1. ``agave`` (or ``agave.exe``) on the system ``PATH``
      2. Standard install locations for the current OS

    Returns the absolute path to the executable, or ``None`` if not found.
    """
    # 1. Check PATH first (works on all platforms).
    on_path = shutil.which("agave")
    if on_path:
        return on_path

    # 2. Platform-specific standard install locations.
    candidates: List[str] = []
    if sys.platform == "win32":
        # Versioned install directories of the form:
        # "Program Files\\AGAVE #.#.#\\agave-install"
        for program_files in (
            os.environ.get("ProgramFiles", "C:\\Program Files"),
            os.environ.get("ProgramFiles(x86)", ""),
        ):
            if not program_files or not os.path.isdir(program_files):
                continue
            try:
                possible = find_matching_subdirectories(
                    program_files, r"AGAVE [0-9]+\.[0-9]+\.[0-9]+"
                )
            except OSError:
                continue
            # If multiple versions, prefer the highest-sorted one.
            for p in sorted(possible):
                candidates.append(os.path.join(p, "agave-install", "agave.exe"))
        candidates.reverse()
    elif sys.platform == "darwin":
        candidates.extend(
            [
                "/Applications/agave.app/Contents/MacOS/agave",
                os.path.expanduser("~/Applications/agave.app/Contents/MacOS/agave"),
            ]
        )
    elif sys.platform.startswith("linux"):
        candidates.extend(
            [
                "/usr/local/bin/agave",
                "/usr/bin/agave",
                "/opt/agave/agave",
                os.path.expanduser("~/.local/bin/agave"),
                os.path.expanduser("~/agave/build/agave"),
            ]
        )
    else:
        return None

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def resolve_agave_path(agave_path: Optional[str] = None) -> str:
    """
    Return a usable AGAVE executable path.

    If ``agave_path`` is given, it is validated and returned. Otherwise
    :func:`guess_agave_path` is consulted. Raises :class:`RuntimeError` if
    no executable can be located, and :class:`FileNotFoundError` if the
    given path does not exist.
    """
    path = agave_path if agave_path else guess_agave_path()
    if not path:
        raise RuntimeError(
            "AGAVE is not running and no installed executable could be "
            "located. Pass agave_path=... or install AGAVE."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"AGAVE executable not found at {path}")
    return path


def launch_agave_process(path: str, port: int = 1235) -> "subprocess.Popen":
    """
    Spawn AGAVE in server mode on the given ``port``.

    Raises :class:`RuntimeError` if the process cannot be started.
    """
    try:
        return subprocess.Popen([path, "--server", f"--port={port}"])
    except OSError as e:
        raise RuntimeError(f"Error launching AGAVE from {path}: {e}") from e


def wait_for_connection(
    connect_fn,
    process: "subprocess.Popen",
    retries: int = 10,
    retry_delay: float = 1.0,
):
    """
    Repeatedly call ``connect_fn()`` until it succeeds or ``retries`` is
    exhausted, monitoring ``process`` for unexpected termination between
    attempts. Returns the value returned by ``connect_fn`` on success.

    On failure the process is terminated and a :class:`RuntimeError` is
    raised describing the last connection error.
    """
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        if process.poll() is not None:
            raise RuntimeError(
                "AGAVE process exited unexpectedly during startup "
                f"(exit code {process.returncode})."
            )
        try:
            return connect_fn()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(retry_delay)

    # All retries exhausted — clean up.
    process.terminate()
    process.wait()
    raise RuntimeError(
        f"Could not connect to AGAVE after {retries} attempts: {last_err}"
    )
