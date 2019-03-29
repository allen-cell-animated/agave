For windows: make sure you are in an environment where vsvarsall has been run, e.g. a "VS2017 x64 Native Tools Command Prompt"

Install CUDA.
Use vcpkg to install boost, tiff, glm.

* for CUDA 9, need to use at least boost 1.65-1 due to an issue with __CUDACC_VER__ .
```
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=D:\vcpkg\scripts\buildsystems\vcpkg.cmake -G "Visual Studio 15 2017 Win64" -DVCPKG_TARGET_TRIPLET=x64-windows -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 5.1.1" ..
cmake --build .
```

For Mac: (using macports)

```
sudo port install boost
sudo port install glm
sudo port install qt5
mkdir build
cd build
cmake ..
make
sudo make install
sudo /opt/local/libexec/qt5/bin/macdeployqt agave-desktop.app -libpath=/opt/local/lib
# sudo cpack -G DragNDrop CPackConfig.cmake
```
OR
```
cmake -G Xcode ..
```

For linux:

* sudo apt install libboost-all-dev
* ensure cuda is installed properly according to http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

* https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux

* current Qt is available via: https://launchpad.net/~beineri (tested with Qt 5.11.2 and 5.9.x)
```
source /opt/qt59/bin/qt59-env.sh # sets QTDIR env var
mkdir build
cd build
~/cmake-3.10.2-Linux-x86_64/bin/cmake -DCMAKE_BUILD_TYPE=Release -DOptiX_INSTALL_DIR=~/NVIDIA-OptiX-SDK-5.1.0-linux64 ..
make
make install
```

# add optix to lib paths for running.
LD_LIBRARY_PATH=~/NVIDIA-OptiX-SDK-5.1.0-linux64/lib64:$LD_LIBRARY_PATH
cd Release/bin
./qtomeapp 
./websockerserverapp

