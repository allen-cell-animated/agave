# Contributing to Allen Institute for Cell Science Open Source

Thank you for your interest in contributing to this Allen Institute for Cell Science open source project! This document is
a set of guidelines to help you contribute to this project.

## Code of Conduct

By participating in this project, you are expected to uphold our [Code of
Conduct][code_of_conduct].

[code_of_conduct]: CODE_OF_CONDUCT.md

## Project Documentation

The `README` in the root of the repository should contain or link to
project documentation. If you cannot find the documentation you're
looking for, please file a GitHub issue with details of what
you'd like to see documented.

The code is currently organized into a few main sections:

1. agave_app is the Qt front end of the application
2. renderlib is the code responsible for dealing with volume images and rendering them
3. agave_pyclient provides a Python package for sending commands to AGAVE when run in server mode

## How to Contribute

Typical steps to contribute:

1. Fork the repo on GitHub.

2. Create a branch and make your edits on your branch, pushing back to your fork.

3. Ensure that your changes are working, pass any linting and tests in the project. Add tests and documentation as needed.

4. Submit a pull request to merge your fork's branch into this repository, via GitHub.

## Questions or Thoughts?

Talk to us on [one of our community forums][community].

[community]: https://forum.allencell.org/

## Releases

on main branch:

1. update the version number in the "project()" in CMakeLists.txt

2. cd agave_pyclient

3. bump2version --current-version oldVersion --new-version newVersion --tag

4. upon push, github actions will automatically build the release.
