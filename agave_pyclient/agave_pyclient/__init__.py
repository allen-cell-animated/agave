# -*- coding: utf-8 -*-

"""Top-level package for agave_pyclient."""

__author__ = "Daniel Toloudis, Allen Institute for Cell Science"
__email__ = "danielt@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "1.1.0"


def get_module_version() -> str:
    return __version__


from .agave import AgaveRenderer  # noqa: F401
