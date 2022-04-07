AGAVE : Advanced GPU Accelerated Volume Explorer

Agave is a desktop application for viewing 16-bit unsigned multichannel ome-tiff and Zeiss .czi files.

The code is currently organized into two main sections:

1. agave_app is the Qt front end of the application
2. renderlib is the code responsible for dealing with volume images and rendering them

How to build from source:

After cloning this repo, initialize the submodules, which contain a couple of dependency libraries:

```
git submodule update --init
```

For WINDOWS:
Make sure you are in an environment where vsvarsall has been run, e.g. a "VS2019 x64 Native Tools Command Prompt"

Install Qt LTS 5.15.2.
In your favorite Python virtual environment:

```
pip install aqtinstall
aqt install --outputdir C:\Qt 5.15.2 windows desktop win64_msvc2019_64
```

Use vcpkg (must use target triplet x64-windows) to install the following:
```
vcpkg install boost-log boost-filesystem glm zlib libjpeg-turbo liblzma tiff --triplet x64-windows
```

```
mkdir build
cd build
# (vs 2019)
cmake -DCMAKE_TOOLCHAIN_FILE=D:\vcpkg\scripts\buildsystems\vcpkg.cmake -G "Visual Studio 16 2019" -A x64 -DVCPKG_TARGET_TRIPLET=x64-windows ..
cmake --build .
```

For MAC OS: (using homebrew)

```
# Install Qt. In your favorite Python virtual environment:
pip install aqtinstall
aqt install --outputdir ~/Qt 5.15.2 mac desktop
export Qt_DIR=~/Qt/5.15.2/clang_64
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

For LINUX:

Install Qt 5.15.2 in your directory of choice and tell the build where to find it.
In your favorite Python virtual environment:

```
pip install aqtinstall
aqt install --outputdir ~/Qt 5.15.2 linux desktop
# the next line is needed for CMake
export Qt5_DIR=~/Qt/5.15.2/gcc_64
```

- sudo apt install libboost-all-dev
- sudo apt install libtiff-dev
- sudo apt install libglm-dev
- sudo apt install libgl1-mesa-dev
- sudo apt install libegl1-mesa-dev

```
mkdir build
cd build
cmake ..
make
```

Versioned Releases

Use tbump (https://github.com/dmerejkowsky/tbump).  See the tbump.toml file which shows all the files that contain necessary version info.

Just run 
```
tbump major.minor.patch --dry-run
```
and if everything looks ok
```
tbump major.minor.patch
```
or, to do the git steps manually:
```
tbump major.minor.patch --only-patch
```
