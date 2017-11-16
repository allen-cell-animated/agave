cmake_minimum_required(VERSION 3.6)

project(qtome)

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
	set(QT_DIR C:/Qt/5.9.1/msvc2015_64)
else (WIN32)
	# linux: QTDIR = /opt/qt59
endif (WIN32)

set(zlib_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/zlib-1.2.11)

set(glm_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glm)

set(CMAKE_PREFIX_PATH ${QT_DIR};${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)

find_package(Qt5Widgets)
find_package(Qt5Core)
find_package(Qt5Gui)
find_package(Qt5Network)
find_package(Qt5WebSockets)
find_package(Qt5Xml)

set(Boost_USE_STATIC_LIBS OFF )
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
#if(MSVC)
  # Debug library suffix.
  set(CMAKE_DEBUG_POSTFIX "d")
  # Require dynamic linking and disable autolinking.
  add_definitions(-DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB)
  # To prevent a lot of spam warnings about standard POSIX functions
  # and unsafe use of the standard library.
  add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
#endif()

set(BOOST_LIBRARYDIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/boost_1_65_1/lib64-msvc-14.0)
find_package(Boost 1.65.1 COMPONENTS chrono log log_setup iostreams thread system filesystem date_time atomic REQUIRED)

set(TIFF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/tiff-4.0.8)
set(TIFF_INCLUDE_DIR ${TIFF_DIR}/include)
set(TIFF_LIBRARY ${TIFF_DIR}/lib/tiff.lib)
find_package(TIFF)

# add_subdirectory(thirdparty)

#####################
# THE STATIC LIB
#####################
find_package(CUDA)
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=sm_50;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=compute_52;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_61,code=sm_61;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_62,code=sm_62;${CUDA_NVCC_FLAGS}")

add_library(renderlib "")
add_subdirectory(renderlib)
target_link_libraries(renderlib
	Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network Qt5::WebSockets Qt5::Xml 
	renderlib_cuda
	tiff
)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

######################
# THE FRONT END QT EXE
######################
add_executable(qtomeapp "")
add_subdirectory(qtome)
target_link_libraries(qtomeapp 
	renderlib 
	Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network Qt5::WebSockets Qt5::Xml
	${TIFF_LIBRARIES}
	Boost::iostreams Boost::log Boost::log_setup Boost::thread Boost::system Boost::filesystem Boost::date_time Boost::chrono Boost::atomic
)
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
set_target_properties(qtomeapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}/dist")
set_target_properties(qtomeapp PROPERTIES VS_STARTUP_PROJECT qtome)
install(TARGETS qtomeapp RUNTIME DESTINATION dist)
install(FILES 
	$<TARGET_FILE:Qt5::Widgets>
	$<TARGET_FILE:Qt5::Core>
	$<TARGET_FILE:Qt5::Gui>
	$<TARGET_FILE:Qt5::Xml>
	${TIFF_DIR}/bin/tiff.dll	
	DESTINATION dist
)
install(DIRECTORY ${BOOST_LIBRARYDIR}/ DESTINATION dist
	FILES_MATCHING PATTERN "*.dll"
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/qtome/ome-qtwidgets-icon/actions DESTINATION dist)


#######################################
# THE FRONT END QT WEBSOCKET SERVER EXE
#######################################
add_executable(websocketserver "")
add_subdirectory(websocketserver)
target_link_libraries(websocketserver 
	renderlib 
	Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network Qt5::WebSockets Qt5::Xml
	Boost::iostreams Boost::log Boost::log_setup Boost::thread Boost::system Boost::filesystem Boost::date_time Boost::chrono Boost::atomic
)
install(TARGETS websocketserver RUNTIME DESTINATION dist)

