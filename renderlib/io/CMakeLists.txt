target_include_directories(renderlib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
)

target_sources(renderlib PRIVATE
"${CMAKE_CURRENT_SOURCE_DIR}/FileReader.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReader.h"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderCCP4.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderCCP4.h"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderCzi.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderCzi.h"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderTIFF.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderTIFF.h"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderZarr.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderZarr.h"
)

# libCZI dependency for renderlib
# add_compile_definitions(_LIBCZISTATICLIB)
set(LIBCZI_DO_NOT_SET_MSVC_RUNTIME_LIBRARY ON)
set(LIBCZI_BUILD_UNITTESTS OFF)
set(LIBCZI_BUILD_DYNLIB ON)
set(LIBCZI_BUILD_CZICMD OFF)
add_subdirectory(libCZI)

# end libczi dependency

# zstd dependency for tensorstore / riegeli
FetchContent_Declare(
	zstd
	URL "https://github.com/facebook/zstd/archive/v1.5.5.tar.gz"
	URL_HASH SHA256=98e9c3d949d1b924e28e01eccb7deed865eefebf25c2f21c702e5cd5b63b85e1
)

# Additional FetchContent_Declare calls as needed...
FetchContent_MakeAvailable(zstd)

# tensorstore dependency for renderlib
set(TENSORSTORE_USE_SYSTEM_TIFF ON)
FetchContent_Declare(
	tensorstore
	URL "https://github.com/google/tensorstore/archive/refs/tags/v0.1.36.tar.gz"
	URL_HASH SHA256=5857582B1B68E5A3F470D2EE8E9A7FA1AD6E6C7E0867C7C69E5A523AE00BC002
)

# Additional FetchContent_Declare calls as needed...
FetchContent_MakeAvailable(tensorstore)

# end tensorstore dependency

target_link_libraries(renderlib
libCZI

tensorstore::tensorstore tensorstore::all_drivers
)

