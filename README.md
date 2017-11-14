For windows: make sure you are in an environment where vsvarsall has been run, e.g. a "VS2015 x64 Native Tools Command Prompt"

put glm in thirdparty/
put ome-files distro in thirdparty/

* for CUDA 9, modify ome/boost/config/compiler/nvcc.h to fix the __CUDACC_VER__ issue.
* boost 1.65-1 fixes the boost/config/compiler/nvcc.h issue and is "compatible" with cuda 9.  File can be overwritten safely.

mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
