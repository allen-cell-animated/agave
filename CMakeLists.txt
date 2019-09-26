cmake_minimum_required(VERSION 3.6)

if (APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12" CACHE STRING "Minimum OS X deployment version" FORCE)
  # set(CMAKE_OSX_SYSROOT "10.12" CACHE STRING "OS X SDK version" FORCE)
endif(APPLE)

project(
  agave 
  VERSION 0.0.4.1
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
  set(QTDIR ~/Qt/5.12.5/gcc_64)
  list(INSERT CMAKE_PREFIX_PATH 0 ${QTDIR})
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif (WIN32)

if (APPLE)
  set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)
  set(CMAKE_MACOSX_RPATH TRUE)
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

################
elseif (APPLE)
################

set(PACKAGE_OSX_TARGET ${CMAKE_OSX_DEPLOYMENT_TARGET})
add_definitions(-DPACKAGE_OSX_TARGET)

install(FILES 
  $<TARGET_FILE:Qt5::Widgets>
  $<TARGET_FILE:Qt5::Core>
  $<TARGET_FILE:Qt5::Gui>
  $<TARGET_FILE:Qt5::Xml>
  $<TARGET_FILE:Qt5::Network>
  $<TARGET_FILE:Qt5::WebSockets>
	DESTINATION ${CMAKE_INSTALL_PREFIX}/agave.app/Contents/Frameworks/
)
install(FILES 
  $<TARGET_FILE:Qt5::QCocoaIntegrationPlugin>
  DESTINATION ${CMAKE_INSTALL_PREFIX}/agave.app/Contents/PlugIns/platforms/
)
install(FILES 
  $<TARGET_FILE:Qt5::QCocoaIntegrationPlugin>
  DESTINATION ${CMAKE_INSTALL_PREFIX}/agave.app/Contents/MacOS/platforms/
)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
  "[Paths]\nPlugins = PlugIns\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/agave.app/Contents/Resources/)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/Info.plist.cmake ${CMAKE_CURRENT_BINARY_DIR}/Info.plist @ONLY)
set_target_properties(qtomeapp PROPERTIES MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_BINARY_DIR}/Info.plist")
# install(FILES "${CMAKE_CURRENT_BINARY_DIR}/Info.plist"
#   DESTINATION ${CMAKE_INSTALL_PREFIX}/agave.app/Contents/)
# set how it shows up in the Info.plist file
# SET(MACOSX_BUNDLE_ICON_FILE JPSvis.icns)
# SET(MACOSX_GUI_ICON JPSvis.icns)
# SET(MACOSX_BUNDLE_COPYRIGHT "Copyright (c) 2019 Allen Institute for Cell Science. All rights reserved.")
# SET(MACOSX_BUNDLE_BUNDLE_NAME "agave")
# SET(MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}")
# SET(MACOSX_BUNDLE_LONG_VERSION_STRING "version ${PROJECT_VERSION}")
# SET(MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
# SET(MACOSX_GUI_COPYRIGHT "Copyright (c) 2019 Allen Institute for Cell Science. All rights reserved.")
# SET(MACOSX_BUNDLE_GUI_IDENTIFIER "www.allencell.org")


# Note Mac specific extension .app
set(APPS "\${CMAKE_INSTALL_PREFIX}/agave.app")

# Directories to look for dependencies
set(DIRS "${CMAKE_BINARY_DIR}")

# Path used for searching by FIND_XXX(), with appropriate suffixes added
if(CMAKE_PREFIX_PATH)
  foreach(dir ${CMAKE_PREFIX_PATH})
    list(APPEND DIRS "${dir}/bin" "${dir}/lib")
  endforeach()
endif()

# Append Qt's lib folder
# list(APPEND DIRS "${QTDIR}/lib")
list(APPEND DIRS "/opt/local/lib")
list(APPEND DIRS "/usr/local/opt/libtiff/lib")

include(InstallRequiredSystemLibraries)

# message(STATUS "APPS: ${APPS}")
# message(STATUS "QT_PLUGINS: ${QT_PLUGINS}")
# message(STATUS "DIRS: ${DIRS}")

# install(CODE "include(BundleUtilities)
#   fixup_bundle(\"${APPS}\" \"${QT_PLUGINS}\" \"${DIRS}\")")

set(CPACK_BINARY_DRAGNDROP ON)
set(CPACK_PACKAGE_NAME "agave")
set(CPACK_PACKAGE_VENDOR "www.allencell.org")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "agave - Advanced GPU Allen Volume Explorer")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
SET(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
SET(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
# set(CPACK_PACKAGE_INSTALL_DIRECTORY "agave_dir")

SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

set(CPACK_GENERATOR "DragNDrop")
set(CPACK_DMG_FORMAT "UDBZ")
set(CPACK_DMG_VOLUME_NAME "${PROJECT_NAME}")
set(CPACK_SYSTEM_NAME "OSX")
set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}")  

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
	DESTINATION ${INSTALLDIR}
)
install(FILES 
  $<TARGET_FILE:Qt5::QXcbIntegrationPlugin>
  DESTINATION ${INSTALLDIR}/platforms/
)

endif (WIN32)
