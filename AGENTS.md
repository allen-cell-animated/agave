# Project Guidelines

## Overview

AGAVE (Advanced GPU Accelerated Volume Explorer) is a scientific 3D volume rendering application for multichannel microscopy data. It consists of a C++/Qt desktop GUI, a core rendering library, a Python client for remote control, and a web client.

## Architecture

- **agave_app/**: Qt 6 desktop GUI — main window, dock widgets, OpenGL viewport (`GLView3D`), and command streaming server
- **renderlib/**: Core rendering engine — file I/O (`io/`), OpenGL shaders (`graphics/`), scene management, command buffer system, and threading utilities
- **agave_pyclient/**: Python package for programmatic control of AGAVE via WebSocket/REST
- **webclient/**: Browser-based client
- **test/**: Catch2 unit tests

File readers implement the `IFileReader` interface. The command buffer system enables serializable command queues for remote control and replay.

## Code Style

- `.clang-format`: Mozilla style, 120-column limit, no automatic include sorting
- **Classes**: PascalCase (`AppScene`, `FileReaderCzi`)
- **Member variables**: `m_` prefix (`m_Film`, `m_Aperture`)
- **Methods**: camelCase, Get/Set for accessors (`getDataMin()`, `SetFilm()`)
- **Constants**: UPPER_CASE (`DEFAULT_PCT_LOW`)
- **Headers**: Use `#pragma once`

## Build and Test

### C++ (CMake)

```bash
# Configure (example for Windows with Qt 6.9.3, in a VS2022 x64 Native Tools Command Prompt)
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=C:\\Users\\%USERNAME%\\source\\repos\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake -G "Ninja Multi-Config" -DVCPKG_TARGET_TRIPLET=x64-windows
# Configure examplefor Linux or macOS with appropriate generator and toolchain settings
cmake -B build -S . -G "Ninja" -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release
# on Windows:
cmake --build build --target install

# Tests run automatically post-build via Catch2. They can be run separately with:
cmake --build build --target agave_test
```

Requires CMake 3.24+, C++17, Qt 6.9.3. Key dependencies: spdlog, GLM, TIFF, libCZI (vendored), tensorstore, pugixml (vendored). See [INSTALL.md](INSTALL.md) for platform-specific setup.

### Python (agave_pyclient)

```bash
pip install -e ".[dev]"
pytest
```

Uses black for formatting and flake8 for linting.

### Version Bumping

Use `tbump` for coordinated version updates across CMakeLists.txt and agave_pyclient.

## Conventions

- Header/implementation split: every `.h` has a corresponding `.cpp`
- Qt MOC is enabled via `AUTOMOC` in CMake — use `Q_OBJECT` macro in QObject subclasses
- Logging uses spdlog macros defined in `renderlib/Logging.h`
- Serialization follows a versioned format (see `Serialize.cpp`, `SerializeV1.cpp`)
- Python client follows PEP 8; C++ follows Mozilla clang-format style
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow
- Adding new commands to the command buffer requires defining a new command class and implementing serialization methods. Whenever a command is added, agave_pyclient and webclient must be updated with the same command id.

### Adding a New Command

A new command requires changes in **four places** across three codebases. All must use the same integer command ID and argument signature.

1. **C++ command definition** (`renderlib/command.h`): Define a data struct (`FooCommandD`) and use the `CMDDECL` macro to declare the command class with its ID, Python name, and argument types.
2. **C++ command implementation** (`renderlib/command.cpp`): Implement `execute()`, `parse()`, `write()`, and `toPythonString()` methods.
3. **C++ dispatch** (`agave_app/commandBuffer.cpp`): Add a `CMD_CASE(FooCommand)` entry to the switch statement.
4. **C++ unit test** (`test/test_commands.cpp`): Add a `SECTION` that round-trips the command through `testcodec` and verifies `toPythonString()` output and data fields.
5. **Python client** (`agave_pyclient/agave_pyclient/commandbuffer.py`): Add the command name, ID, and argument types to the `COMMANDS` dict. Then add a wrapper method in `agave_pyclient/agave_pyclient/agave.py`.
6. **Web client** (`webclient/src/commandbuffer.ts`): Add the command to the `COMMANDS` export. Then add a wrapper method in `webclient/src/agave.ts`.
