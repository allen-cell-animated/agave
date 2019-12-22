AGAVE : Advanced GPU Accelerated Volume Explorer

Agave is a desktop application for viewing 16-bit unsigned multichannel ome-tif files.

The code is currently organized into two sections:

1. qtome is the Qt front end of the application
2. renderlib is the code responsible for dealing with volume images and rendering them

How to build from source:

For windows: make sure you are in an environment where vsvarsall has been run, e.g. a "VS2017 x64 Native Tools Command Prompt"

Use official install of Qt 5.12.5 or greater.
Use vcpkg to install boost, tiff, glm. Make sure the vcpkg target triplet is x64-windows.

```
mkdir build
cd build
# (vs 2019)
cmake -DCMAKE_TOOLCHAIN_FILE=D:\vcpkg\scripts\buildsystems\vcpkg.cmake -G "Visual Studio 16 2019" -A x64 -DVCPKG_TARGET_TRIPLET=x64-windows ..
cmake --build .
```

For Mac: (using homebrew)

```
# use official install of Qt for Mac
# and then:
brew install boost glm libtiff

mkdir build
cd build
cmake ..
make
# after make, you should have a runnable agave.app
# or, to build a redistributable bundle:
sudo make install
```

OR

```
cmake -G Xcode ..
```

For linux:

- sudo apt install libboost-all-dev
- sudo apt install libtiff-dev
- sudo apt install libglm-dev
- sudo apt install libgl1-mesa-dev

- use official Qt 5.12.5 installer for linux and install into default location (~/Qt)

```
mkdir build
cd build
cmake ..
make
```
