For windows: make sure you are in an environment where vsvarsall has been run, e.g. a "VS2015 x64 Native Tools Command Prompt"

Install CUDA.
Use vcpkg to install boost, tiff, and glm.

* for CUDA 9, need to use at least boost 1.65-1 due to an issue with __CUDACC_VER__ .

mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=D:\vcpkg\scripts\buildsystems\vcpkg.cmake -G "Visual Studio 14 2015 Win64" -DVCPKG_TARGET_TRIPLET=x64-windows ..
cmake --build .

For linux:

mkdir build
cd build
BOOST_ROOT=/path/to/boost_1_65_1_build/boost ~/cmake-3.9.6-Linux-x86_64/bin/cmake -DCMAKE_BUILD_TYPE=Debug ..
make
