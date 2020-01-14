cmake_minimum_required(VERSION 3.6)

if (APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12" CACHE STRING "Minimum OS X deployment version" FORCE)
endif(APPLE)

project(
  agave 
  VERSION 0.0.5.0
  LANGUAGES C CXX
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
  set(QTDIR C:/Qt/5.12.5/msvc2017_64)
  list(INSERT CMAKE_PREFIX_PATH 0 ${QTDIR})
elseif (APPLE)
  # mac official Qt install:
  set(QTDIR ~/Qt/5.12.5/clang_64)
  list(INSERT CMAKE_PREFIX_PATH 0 ${QTDIR})
else ()
  # set(QTDIR ~/Qt/5.12.5/gcc_64)
  list(INSERT CMAKE_PREFIX_PATH 0 $ENV{QTDIR})
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif (WIN32)

if (APPLE)
  set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)
  # set(CMAKE_MACOSX_RPATH TRUE)
endif (APPLE)

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

# the following line is a workaround for homebrew boost 1.71 + cmake not finding the boost multithreaded libs
# https://gitlab.kitware.com/cmake/cmake/issues/19714
set(Boost_NO_BOOST_CMAKE 1)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS chrono log log_setup iostreams thread system filesystem date_time atomic REQUIRED)

find_package(TIFF REQUIRED)

# set(glm_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glm)
# find_path(GLM_PATH glm/glm.hpp)
find_package( glm REQUIRED )

######################
# THE COMMON CORE LIBRARIES
######################
add_subdirectory(renderlib)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
set(INSTALLDIR "${CMAKE_PROJECT_NAME}-${PROJECT_VERSION}")

message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "INSTALLDIR: ${INSTALLDIR}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

######################
# THE FRONT END QT EXE
######################
add_subdirectory(qtome)
#######################################
# THE FRONT END QT WEBSOCKET SERVER EXE
#######################################
# currently this is folded into the main agave app with the --server command line arg
# add_subdirectory(websocketserver)

######################
# CPack
######################


# copy all dlls into place.
if (WIN32)
install(FILES 
  ${PROJECT_SOURCE_DIR}/LICENSE.txt
  ${PROJECT_SOURCE_DIR}/HELP.txt
  $<TARGET_FILE:Qt5::Widgets>
  $<TARGET_FILE:Qt5::Core>
  $<TARGET_FILE:Qt5::Gui>
  $<TARGET_FILE:Qt5::Xml>
  $<TARGET_FILE:Qt5::Network>
  $<TARGET_FILE:Qt5::WebSockets>
  # note these are release build dlls.
  ${PYTHON_LIBRARIES}
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_chrono-vc142-mt-x64-1_70.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_date_time-vc142-mt-x64-1_70.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_filesystem-vc142-mt-x64-1_70.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_log-vc142-mt-x64-1_70.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_system-vc142-mt-x64-1_70.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_thread-vc142-mt-x64-1_70.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/jpeg62.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/lzma.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/tiff.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/zlib1.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/zstd.dll
  DESTINATION ${INSTALLDIR}
  CONFIGURATIONS Release
)
install(FILES 
  ${PROJECT_SOURCE_DIR}/LICENSE.txt
  ${PROJECT_SOURCE_DIR}/HELP.txt
  $<TARGET_FILE:Qt5::Widgets>
  $<TARGET_FILE:Qt5::Core>
  $<TARGET_FILE:Qt5::Gui>
  $<TARGET_FILE:Qt5::Xml>
  $<TARGET_FILE:Qt5::Network>
  $<TARGET_FILE:Qt5::WebSockets>
  # debug dlls
  ${PYTHON_LIBRARIES}
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_chrono-vc142-mt-gd-x64-1_70.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_date_time-vc142-mt-gd-x64-1_70.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_filesystem-vc142-mt-gd-x64-1_70.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_log-vc142-mt-gd-x64-1_70.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_system-vc142-mt-gd-x64-1_70.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_thread-vc142-mt-gd-x64-1_70.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/jpeg62.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/lzmad.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/tiffd.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/zlibd1.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/zstdd.dll
  DESTINATION ${INSTALLDIR}
  CONFIGURATIONS Debug
)
install(FILES 
  $<TARGET_FILE:Qt5::QWindowsIntegrationPlugin>
  DESTINATION ${INSTALLDIR}/platforms/
)
install(FILES 
  $<TARGET_FILE:Qt5::QJpegPlugin>
  $<TARGET_FILE:Qt5::QTiffPlugin>
  DESTINATION ${INSTALLDIR}/imageformats/
)

################
elseif (APPLE)
################

set(PACKAGE_OSX_TARGET ${CMAKE_OSX_DEPLOYMENT_TARGET})
add_definitions(-DPACKAGE_OSX_TARGET)

# based on code from CMake's QtDialog/CMakeLists.txt
macro(install_qt5_plugin _qt_plugin_name _qt_plugins_var _prefix)
    get_target_property(_qt_plugin_path "${_qt_plugin_name}" LOCATION)
    if(EXISTS "${_qt_plugin_path}")
        get_filename_component(_qt_plugin_file "${_qt_plugin_path}" NAME)
        get_filename_component(_qt_plugin_type "${_qt_plugin_path}" PATH)
        get_filename_component(_qt_plugin_type "${_qt_plugin_type}" NAME)
        set(_qt_plugin_dest "${_prefix}/PlugIns/${_qt_plugin_type}")
        install(FILES "${_qt_plugin_path}"
            DESTINATION "${_qt_plugin_dest}")
        set(${_qt_plugins_var}
            "${${_qt_plugins_var}};${_qt_plugin_dest}/${_qt_plugin_file}")
    else()
        message(FATAL_ERROR "QT plugin ${_qt_plugin_name} not found")
    endif()
endmacro()


install(FILES 
  $<TARGET_FILE:Qt5::Widgets>
  $<TARGET_FILE:Qt5::Core>
  $<TARGET_FILE:Qt5::Gui>
  $<TARGET_FILE:Qt5::Xml>
  $<TARGET_FILE:Qt5::Network>
  $<TARGET_FILE:Qt5::WebSockets>
  ${PYTHON_LIBRARIES}
	DESTINATION ${CMAKE_INSTALL_PREFIX}/agave.app/Contents/Frameworks/
)


install_qt5_plugin("Qt5::QCocoaIntegrationPlugin" QT_PLUGINS ${CMAKE_INSTALL_PREFIX}agave.app/Contents)
# set(${QT_PLUGINS} 
#             "${${_qt_plugins_var}};${_qt_plugin_dest}/${_qt_plugin_file}")
# install(FILES 
#   $<TARGET_FILE:Qt5::QCocoaIntegrationPlugin>
#   DESTINATION ${CMAKE_INSTALL_PREFIX}/agave.app/Contents/PlugIns/platforms/
# )
install(FILES 
  $<TARGET_FILE:Qt5::QCocoaIntegrationPlugin>
  DESTINATION ${CMAKE_INSTALL_PREFIX}/agave.app/Contents/MacOS/platforms/
)
install(FILES 
  $<TARGET_FILE:Qt5::QJpegPlugin>
  $<TARGET_FILE:Qt5::QTiffPlugin>
  DESTINATION ${CMAKE_INSTALL_PREFIX}/agave.app/Contents/PlugIns/imageformats/
)
install(FILES 
  $<TARGET_FILE:Qt5::QJpegPlugin>
  $<TARGET_FILE:Qt5::QTiffPlugin>
  DESTINATION ${CMAKE_INSTALL_PREFIX}/agave.app/Contents/MacOS/imageformats/
)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
  "[Paths]\nPlugins = PlugIns\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/agave.app/Contents/Resources/)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/Info.plist.cmake ${CMAKE_CURRENT_BINARY_DIR}/Info.plist @ONLY)
set_target_properties(qtomeapp PROPERTIES MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_BINARY_DIR}/Info.plist")

# Note Mac specific extension .app
set(APPS "\${CMAKE_INSTALL_PREFIX}/agave.app")

# Directories to look for dependencies
set(DIRS "${CMAKE_BINARY_DIR}")

# Path used for searching by FIND_XXX(), with appropriate suffixes added
if(CMAKE_PREFIX_PATH)
  foreach(dir ${CMAKE_PREFIX_PATH})
    # note that this should get the necessary Qt lib directories
    list(APPEND DIRS "${dir}/bin" "${dir}/lib")
  endforeach()
endif()

# Append Qt's lib folder
# list(APPEND DIRS "${QTDIR}/lib")
# list(APPEND DIRS "${Qt5Widgets_DIR}/../..")

set(PLUGINS "${QT_PLUGINS}")

include(InstallRequiredSystemLibraries)

message(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
message(STATUS "APPS: ${APPS}")
message(STATUS "PLUGINS: ${PLUGINS}")
message(STATUS "DIRS: ${DIRS}")

install(CODE "include(BundleUtilities)
    fixup_bundle(\${CMAKE_INSTALL_PREFIX}/agave.app \"${PLUGINS}\" \"${DIRS}\" IGNORE_ITEM \"Python\")" COMPONENT Runtime)

set(CPACK_GENERATOR "DragNDrop")
include(CPack)

else() # Linux

install(FILES 
  ${PROJECT_SOURCE_DIR}/LICENSE.txt
  ${PROJECT_SOURCE_DIR}/HELP.txt
  $<TARGET_FILE:Qt5::Widgets>
  $<TARGET_FILE:Qt5::Core>
  $<TARGET_FILE:Qt5::Gui>
  $<TARGET_FILE:Qt5::Xml>
  $<TARGET_FILE:Qt5::Network>
  $<TARGET_FILE:Qt5::WebSockets>
  ${PYTHON_LIBRARIES}
	DESTINATION ${INSTALLDIR}
)
install(FILES 
  $<TARGET_FILE:Qt5::QXcbIntegrationPlugin>
  DESTINATION ${INSTALLDIR}/platforms/
)
install(FILES 
  $<TARGET_FILE:Qt5::QJpegPlugin>
  $<TARGET_FILE:Qt5::QTiffPlugin>
  DESTINATION ${INSTALLDIR}/imageformats/
)

endif (WIN32)
