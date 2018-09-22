cmake_minimum_required(VERSION 3.6)

project(
  qtome 
  VERSION 1.0.0
  LANGUAGES C CXX CUDA
)

# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

# Find the Qt libraries
# In order for find_package to be successful, Qt 5 must be found below the CMAKE_PREFIX_PATH, 
# or the Qt5<Module>_DIR must be set in the CMake cache to the location of the Qt5WidgetsConfig.cmake 
# file. The easiest way to use CMake is to set the CMAKE_PREFIX_PATH environment variable to
# the install prefix of Qt 5.
if (WIN32)
  set(QTDIR C:/Qt/5.11.1/msvc2017_64)
  list(INSERT CMAKE_PREFIX_PATH 0 ${QTDIR})
else (WIN32)
  # linux: QTDIR = /opt/qt59
endif (WIN32)

find_package(Qt5Widgets)
find_package(Qt5Core)
find_package(Qt5Gui)
find_package(Qt5Network)
find_package(Qt5WebSockets)
find_package(Qt5Xml)

#if(MSVC)
  # Debug library suffix.
  set(CMAKE_DEBUG_POSTFIX "d")
  # To prevent a lot of spam warnings about standard POSIX functions
  # and unsafe use of the standard library.
  add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
#endif()

find_package(Boost COMPONENTS chrono log log_setup iostreams thread system filesystem date_time atomic REQUIRED)

find_package(TIFF REQUIRED)

find_package(OptiX REQUIRED)
find_package(assimp REQUIRED)

set(CMAKE_CUDA_FLAGS
  # Pascal target
  "-gencode=arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets"
  # "-arch=sm_30 \
  #   -gencode=arch=compute_30,code=sm_30 \
  #   -gencode=arch=compute_50,code=sm_50 \
  #   -gencode=arch=compute_52,code=sm_52 \
  #   -gencode=arch=compute_60,code=sm_60 \
  #   -gencode=arch=compute_61,code=sm_61 \
  #   -gencode=arch=compute_62,code=sm_62 \
  #   -gencode=arch=compute_70,code=sm_70 \
  #   -gencode=arch=compute_70,code=compute_70 \
  #   -Wno-deprecated-gpu-targets"
)


set(glm_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glm)

######################
# THE COMMON CORE LIBRARIES
######################
add_subdirectory(renderlib)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})

######################
# THE FRONT END QT EXE
######################
add_subdirectory(qtome)
#######################################
# THE FRONT END QT WEBSOCKET SERVER EXE
#######################################
add_subdirectory(websocketserver)

# copy all dlls into place.
if (WIN32)
install(FILES 
	$<TARGET_FILE:Qt5::Widgets>
	$<TARGET_FILE:Qt5::Core>
	$<TARGET_FILE:Qt5::Gui>
	$<TARGET_FILE:Qt5::Xml>
	$<TARGET_FILE:Qt5::Network>
  $<TARGET_FILE:Qt5::WebSockets>
  ${optix_DLL}
  # note these are release build dlls.
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/assimp-vc140-mt.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_chrono-vc141-mt-x64-1_67.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_date_time-vc141-mt-x64-1_67.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_filesystem-vc141-mt-x64-1_67.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_log-vc141-mt-x64-1_67.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_system-vc141-mt-x64-1_67.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_thread-vc141-mt-x64-1_67.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/jpeg62.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/lzma.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/tiff.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/zlib1.dll
  # debug dlls
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/assimp-vc140-mtd.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_chrono-vc141-mt-gd-x64-1_67.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_date_time-vc141-mt-gd-x64-1_67.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_filesystem-vc141-mt-gd-x64-1_67.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_log-vc141-mt-gd-x64-1_67.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_system-vc141-mt-gd-x64-1_67.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_thread-vc141-mt-gd-x64-1_67.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/jpeg62.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/lzma.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/tiffd.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/zlibd1.dll
	DESTINATION bin
)
endif (WIN32)



