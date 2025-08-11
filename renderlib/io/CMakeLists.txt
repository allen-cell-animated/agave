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
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderTIFF.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderTIFF.h"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderZarr.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/FileReaderZarr.h"
)

# libCZI dependency for renderlib
# add_compile_definitions(_LIBCZISTATICLIB)
set(LIBCZI_DO_NOT_SET_MSVC_RUNTIME_LIBRARY ON)
if(WIN32)
  set(LIBCZI_BUILD_PREFER_EXTERNALPACKAGE_EIGEN3 ON)
endif(WIN32)
set(LIBCZI_BUILD_UNITTESTS OFF)
set(LIBCZI_BUILD_DYNLIB ON)
set(LIBCZI_BUILD_CZICMD OFF)
add_subdirectory(libCZI)

# end libczi dependency

# tensorstore dependency for renderlib
set(TENSORSTORE_USE_SYSTEM_TIFF ON)
FetchContent_Declare(
	tensorstore
        # URL "https://github.com/google/tensorstore/archive/refs/tags/v0.1.71.tar.gz"
        # URL_HASH SHA256=997ae179a3e6203c745afe9dfd5a1594ce5cf1738ebfafd3031f5f23211929f2
        URL "https://github.com/google/tensorstore/tarball/9e1feb7215ee876d3b62b8d84c03c9c2a1eec0c2"
        URL_HASH SHA256=baeb9e1eafe1d8244d37c614711a8125467404801274e35979ba2896c8abdf29
)

# Additional FetchContent_Declare calls as needed...
FetchContent_MakeAvailable(tensorstore)
# end tensorstore dependency

target_link_libraries(renderlib
libCZI

tensorstore::tensorstore tensorstore::all_drivers
)

