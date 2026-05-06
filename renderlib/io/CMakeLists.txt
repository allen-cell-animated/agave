target_include_directories(renderlib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
)

target_sources(renderlib PRIVATE
"${CMAKE_CURRENT_SOURCE_DIR}/ConvertChannelData.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/ConvertChannelData.h"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReader.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReader.h"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderCCP4.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderCCP4.h"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderCzi.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderCzi.h"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderImageSequence.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderImageSequence.h"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderND2.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderND2.h"
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

# tensorstore dependency for renderlib
set(TENSORSTORE_USE_SYSTEM_TIFF ON)
if(APPLE)
  set(TENSORSTORE_USE_SYSTEM_ZLIB ON)
endif(APPLE)
FetchContent_Declare(
	tensorstore
        URL "https://github.com/google/tensorstore/archive/refs/tags/v0.1.78.tar.gz"
        URL_HASH SHA256=f59667a32357b8cc0c752429927ad97654f6a67c7d3d62b9efea902c6798d473
        # URL "https://github.com/google/tensorstore/tarball/1461a5247186a84f5be7b632309ca0f5bf05a55e"
        # URL_HASH SHA256=1ac9ac2c3a24d0c25f049d289968b4e3944a798a9466b84d80a964b74a2d018a
)

FetchContent_MakeAvailable(tensorstore)
# end tensorstore dependency

# zlib + lz4 dependencies for the ND2 reader (frame decompression).
# tensorstore already pulls zlib in; on macOS it uses the system zlib. We
# resolve the include path / link via find_package, but use the variable
# form (not ZLIB::ZLIB) because the imported target is not always exposed
# in this scope after tensorstore's nested configuration.
find_package(ZLIB REQUIRED)
# tensorstore (via its blosc dependency) already builds LZ4 and exposes
# the LZ4::LZ4 target with sources at ${lz4_SOURCE_DIR}/lib.

target_link_libraries(renderlib
libCZI

tensorstore::tensorstore tensorstore::all_drivers
${ZLIB_LIBRARIES}
LZ4::LZ4
)

target_include_directories(renderlib PRIVATE
  "${lz4_SOURCE_DIR}/lib"
  ${ZLIB_INCLUDE_DIRS}
)

