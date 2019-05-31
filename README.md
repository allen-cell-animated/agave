AGAVE : Advanced GPU Accelerated Volume Explorer

Agave is a desktop application for viewing 16-bit unsigned multichannel ome-tif files.

The code is currently organized into two sections: 
1. qtome is the Qt front end of the application
2. renderlib is the code responsible for dealing with volume images and rendering them

How to build from source:

For windows: make sure you are in an environment where vsvarsall has been run, e.g. a "VS2017 x64 Native Tools Command Prompt"

Use official install of Qt 5.12.2 or greater. 
Use vcpkg to install boost, tiff, glm.

```
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=D:\vcpkg\scripts\buildsystems\vcpkg.cmake -G "Visual Studio 15 2017 Win64" -DVCPKG_TARGET_TRIPLET=x64-windows ..
cmake --build .
```

For Mac: (using macports)

```
# use official install of Qt for Mac
sudo port install boost
sudo port install glm
sudo port install tiff
mkdir build
cd build
cmake ..
make
sudo make install
sudo $HOME/Qt/5.12.3/clang_64/bin/macdeployqt agave.app -libpath=/opt/local/lib -always-overwrite -appstore-compliant
```

OR

```
cmake -G Xcode ..
```

For linux:

- sudo apt install libboost-all-dev

- current Qt is available via: https://launchpad.net/~beineri (tested with Qt 5.11.2 and 5.9.x)

```
source /opt/qt59/bin/qt59-env.sh # sets QTDIR env var
mkdir build
cd build
~/cmake-3.10.2-Linux-x86_64/bin/cmake -DCMAKE_BUILD_TYPE=Release ..
make
make install
```

cd Release/bin
./qtomeapp
./websockerserverapp
