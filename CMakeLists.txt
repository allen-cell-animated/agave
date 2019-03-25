cmake_minimum_required(VERSION 3.6)

if (APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13")
endif(APPLE)

project(
  agave 
  VERSION 0.0.1.0
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
  set(QTDIR C:/Qt/5.11.2/msvc2017_64)
  list(INSERT CMAKE_PREFIX_PATH 0 ${QTDIR})
else (WIN32)
  # linux: QTDIR = /opt/qt59
  set(QTDIR /opt/local/libexec/qt5)
  list(INSERT CMAKE_PREFIX_PATH 0 ${QTDIR})
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

if (APPLE)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}.app/Contents)
  set(INSTALLDIR "${CMAKE_INSTALL_PREFIX}")
else ()
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
  set(INSTALLDIR "${CMAKE_PROJECT_NAME}-${PROJECT_VERSION}")
endif(APPLE)
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
add_subdirectory(websocketserver)

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
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_chrono-vc141-mt-x64-1_68.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_date_time-vc141-mt-x64-1_68.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_filesystem-vc141-mt-x64-1_68.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_log-vc141-mt-x64-1_68.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_system-vc141-mt-x64-1_68.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/boost_thread-vc141-mt-x64-1_68.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/jpeg62.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/lzma.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/tiff.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/zlib1.dll
	${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/zstd.dll
  # debug dlls
	# ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_chrono-vc141-mt-gd-x64-1_68.dll
	# ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_date_time-vc141-mt-gd-x64-1_68.dll
	# ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_filesystem-vc141-mt-gd-x64-1_68.dll
	# ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_log-vc141-mt-gd-x64-1_68.dll
	# ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_system-vc141-mt-gd-x64-1_68.dll
	# ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/boost_thread-vc141-mt-gd-x64-1_68.dll
	# ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/jpeg62.dll
	# ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/lzma.dll
	# ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/tiffd.dll
	# ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/zlibd1.dll
	# ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/zstdd.dll
	DESTINATION ${INSTALLDIR}
)
install(FILES 
  $<TARGET_FILE:Qt5::QWindowsIntegrationPlugin>
  DESTINATION ${INSTALLDIR}/platforms/
)

# Note Mac specific extension .app
set(APPS ${INSTALLDIR})

# Directories to look for dependencies
set(DIRS "${CMAKE_BINARY_DIR}")

# Path used for searching by FIND_XXX(), with appropriate suffixes added
if(CMAKE_PREFIX_PATH)
  foreach(dir ${CMAKE_PREFIX_PATH})
    list(APPEND DIRS "${dir}/bin" "${dir}/lib")
  endforeach()
endif()

# Append Qt's lib folder which is two levels above Qt5Widgets_DIR
list(APPEND DIRS "${Qt5Widgets_DIR}/../..")
list(APPEND DIRS "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")

include(InstallRequiredSystemLibraries)

message(STATUS "APPS: ${APPS}")
message(STATUS "QT_PLUGINS: ${QT_PLUGINS}")
message(STATUS "DIRS: ${DIRS}")

install(CODE "include(BundleUtilities)
  fixup_bundle(\"${APPS}\" \"${QT_PLUGINS}\" \"${DIRS}\")")

elseif (APPLE)

set(PACKAGE_OSX_TARGET ${CMAKE_OSX_DEPLOYMENT_TARGET})
add_definitions(-DPACKAGE_OSX_TARGET)

install(FILES 
  # ${PROJECT_SOURCE_DIR}/LICENSE.txt
  # ${PROJECT_SOURCE_DIR}/HELP.txt
  $<TARGET_FILE:Qt5::Widgets>
  $<TARGET_FILE:Qt5::Core>
  $<TARGET_FILE:Qt5::Gui>
  $<TARGET_FILE:Qt5::Xml>
  $<TARGET_FILE:Qt5::Network>
  $<TARGET_FILE:Qt5::WebSockets>
	DESTINATION ./Frameworks/
)
install(FILES 
  $<TARGET_FILE:Qt5::QCocoaIntegrationPlugin>
  DESTINATION ./PlugIns/platforms/
)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
  "[Paths]\nPlugins = PlugIns\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
  DESTINATION ./Resources/)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/Info.plist.cmake ${CMAKE_CURRENT_BINARY_DIR}/Info.plist @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/Info.plist"
  DESTINATION .)

# Note Mac specific extension .app
set(APPS ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}.app)

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

message(STATUS "APPS: ${APPS}")
message(STATUS "QT_PLUGINS: ${QT_PLUGINS}")
message(STATUS "DIRS: ${DIRS}")

install(CODE "include(BundleUtilities)
  fixup_bundle(\"${APPS}\" \"${QT_PLUGINS}\" \"${DIRS}\")")

set(CPACK_GENERATOR "DragNDrop")
set(CPACK_DMG_FORMAT "UDBZ")
set(CPACK_DMG_VOLUME_NAME "${PROJECT_NAME}")
set(CPACK_SYSTEM_NAME "OSX")
# SET(CMAKE_INSTALL_UCRT_LIBRARIES TRUE)
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AGAVE is a volume viewer using cinematic rendering techniques with realistic lighting")
SET(CPACK_PACKAGE_VENDOR "AICS")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
SET(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
SET(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "agave")
# SET(CPACK_STRIP_FILES ON)
SET(CPACK_PACKAGE_EXECUTABLES "agave-desktop" "Agave")
# SET(CPACK_RPM_PACKAGE_LICENSE "GPLv2+")
# SET(CPACK_RPM_PACKAGE_GROUP "Applications/Multimedia")
# SET(CPACK_RPM_PACKAGE_URL "http://allencell.org/")
# SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_VENDOR} <animated-cell@allencell.org>")
# SET(CPACK_DEBIAN_PACKAGE_SECTION "science") # or graphics?
# SET(CPACK_DEBIAN_PACKAGE_VERSION "${VERSION}+deb1")
# SET(CPACK_DEBIAN_PACKAGE_HOMEPAGE "${CPACK_RPM_PACKAGE_URL}")
# SET(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
include(CPack)

endif (WIN32)
