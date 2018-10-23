For windows: make sure you are in an environment where vsvarsall has been run, e.g. a "VS2015 x64 Native Tools Command Prompt"

Install CUDA.
Use vcpkg to install boost, tiff, and glm.

* for CUDA 9, need to use at least boost 1.65-1 due to an issue with __CUDACC_VER__ .
```
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=D:\vcpkg\scripts\buildsystems\vcpkg.cmake -G "Visual Studio 14 2015 Win64" -DVCPKG_TARGET_TRIPLET=x64-windows -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 5.1.0" ..                                                                                                                                                                                               
cmake --build .
```

For linux:

* sudo apt install libboost-all-dev
* libassimp-dev
* ensure cuda is installed properly according to http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
```
source /opt/qt59/bin/qt59-env.sh # sets QTDIR env var
mkdir build
cd build
~/cmake-3.10.2-Linux-x86_64/bin/cmake -DCMAKE_BUILD_TYPE=Release -DOptiX_INSTALL_DIR=~/NVIDIA-OptiX-SDK-5.1.0-linux64 ..
make
make install
```

# add boost and optix to lib paths for running.
LD_LIBRARY_PATH=~/NVIDIA-OptiX-SDK-5.1.0-linux64/lib64:$LD_LIBRARY_PATH
cd Release/bin
./qtomeapp 
./websockerserverapp

