For windows: make sure you are in an environment where vsvarsall has been run, e.g. a "VS2015 x64 Native Tools Command Prompt"

put glm in thirdparty/
put ome-files distro in thirdparty/

mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
