cmake_minimum_required(VERSION 3.16)

if(APPLE)
  set(ENV{MACOSX_DEPLOYMENT_TARGET} "10.15")
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version" FORCE)
endif(APPLE)

if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

cmake_policy(SET CMP0091 NEW) # enable new "MSVC runtime library selection" (https://cmake.org/cmake/help/latest/variable/CMAKE_MSVC_RUNTIME_LIBRARY.html)

project(
  agave
  VERSION 1.4.1
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)

IF(WIN32)
ELSE()
  # most of this is for libCZI (?):
  # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++17 -fPIC -O0 -g -D_FILE_OFFSET_BITS=64 -fvisibility=hidden")
  # SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__ANSI__ -fPIC -D_FILE_OFFSET_BITS=64")
  add_compile_definitions(LINUXENV)
ENDIF(WIN32)

# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
)

# Find includes in corresponding build directories
# set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Find the Qt libraries
# In order for find_package to be successful, Qt 5 must be found below the CMAKE_PREFIX_PATH,
# or the Qt5<Module>_DIR must be set in the CMake cache to the location of the Qt5WidgetsConfig.cmake
# file. The easiest way to use CMake is to set the CMAKE_PREFIX_PATH environment variable to
# the install prefix of Qt 5.
if(DEFINED ENV{Qt5_DIR})
  list(INSERT CMAKE_PREFIX_PATH 0 $ENV{Qt5_DIR})
endif()

set(AGAVE_QT_VERSION 5.15.2)

if(WIN32)
  set(GUESS_Qt5_DIR C:/Qt/${AGAVE_QT_VERSION}/msvc2019_64 CACHE STRING "Qt5 directory")

# set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
elseif(APPLE)
  set(GUESS_Qt5_DIR ~/Qt/${AGAVE_QT_VERSION}/clang_64)
else()
  set(GUESS_Qt5_DIR /qt/${AGAVE_QT_VERSION}/gcc_64)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif(WIN32)

list(INSERT CMAKE_PREFIX_PATH 0 ${GUESS_Qt5_DIR})

if(APPLE)
  # homebrew
  list(APPEND CMAKE_LIBRARY_PATH /usr/local/lib /usr/local/opt/icu4c/lib)
  list(APPEND CMAKE_INCLUDE_PATH /usr/local/include)

  # set(CMAKE_MACOSX_RPATH TRUE)
  add_compile_options("-I/usr/local/opt/icu4c/include")
  add_link_options("-L/usr/local/opt/icu4c/lib")
endif(APPLE)

find_package(Qt5 COMPONENTS Widgets Core Gui Network WebSockets Xml REQUIRED)

find_package(spdlog REQUIRED)

# if(MSVC)
# Debug library suffix.
# set(CMAKE_DEBUG_POSTFIX "d")
# To prevent a lot of spam warnings about standard POSIX functions
# and unsafe use of the standard library.
add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)

# endif()

# set(glm_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glm)
if(APPLE)
  # On MacOS, newer homebrew glm is broken with respect to cmake.
  # Revisit in a future version to try to remove this conditional.
  find_path(GLM_PATH glm/glm.hpp)
else()
  find_package(glm REQUIRED)
endif(APPLE)

find_package(TIFF)

find_package(OpenGL)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})
else()
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# #####################
# THE COMMON CORE LIBRARIES
# #####################
add_subdirectory(renderlib)

set(INSTALLDIR "${CMAKE_PROJECT_NAME}-install")

message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "INSTALLDIR: ${INSTALLDIR}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# #####################
# UNIT TESTING
# #####################
add_subdirectory(test)

# #####################
# THE FRONT END QT EXE
# #####################
add_subdirectory(agave_app)

# #####################
# Documentation
# #####################
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/conf.py.cmake ${CMAKE_SOURCE_DIR}/docs/conf.py @ONLY)

# #####################
# CPack
# #####################

# copy all dlls into place.
if(WIN32)
  # assuming vcpkg has done some work already to get dlls in here
  install(
    DIRECTORY ${CMAKE_BINARY_DIR}/Release/
    DESTINATION ${INSTALLDIR}
    CONFIGURATIONS Release
    FILES_MATCHING PATTERN "*.dll"
  )
  install(
    DIRECTORY ${CMAKE_BINARY_DIR}/Debug/
    DESTINATION ${INSTALLDIR}
    CONFIGURATIONS Debug
    FILES_MATCHING PATTERN "*.dll"
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
    $<TARGET_FILE:TIFF::TIFF>
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
    $<TARGET_FILE:TIFF::TIFF>
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
  install(FILES
    $<TARGET_FILE:Qt5::QWindowsVistaStylePlugin>
    DESTINATION ${INSTALLDIR}/styles/
  )

# ###############
elseif(APPLE)
  # ###############
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
    $<TARGET_FILE:TIFF::TIFF>
    DESTINATION ${CMAKE_INSTALL_PREFIX}/agave.app/Contents/Frameworks/
  )

  # calls cmake install command and also adds to the QT_PLUGINS list for use below
  install_qt5_plugin("Qt5::QCocoaIntegrationPlugin" QT_PLUGINS ${CMAKE_INSTALL_PREFIX}/agave.app/Contents)
  install_qt5_plugin("Qt5::QJpegPlugin" QT_PLUGINS ${CMAKE_INSTALL_PREFIX}/agave.app/Contents)
  install_qt5_plugin("Qt5::QTiffPlugin" QT_PLUGINS ${CMAKE_INSTALL_PREFIX}/agave.app/Contents)
  install_qt5_plugin("Qt5::QMacStylePlugin" QT_PLUGINS ${CMAKE_INSTALL_PREFIX}/agave.app/Contents)

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
    "[Paths]\nPlugins = PlugIns\n")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
    DESTINATION ${CMAKE_INSTALL_PREFIX}/agave.app/Contents/Resources/)

  CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/Info.plist.cmake ${CMAKE_CURRENT_BINARY_DIR}/Info.plist @ONLY)
  set_target_properties(agaveapp PROPERTIES MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_BINARY_DIR}/Info.plist")

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

  # homebrew lib path
  list(APPEND DIRS "/usr/local/lib")

  # Append Qt's lib folder
  # list(APPEND DIRS "${QTDIR}/lib")
  # list(APPEND DIRS "${Qt5Widgets_DIR}/../..")
  set(PLUGINS "${QT_PLUGINS}")

  include(InstallRequiredSystemLibraries)

  message(STATUS "APPS: ${APPS}")
  message(STATUS "PLUGINS: ${PLUGINS}")
  message(STATUS "DIRS: ${DIRS}")

  install(CODE "include(BundleUtilities)
    fixup_bundle(\${CMAKE_INSTALL_PREFIX}/agave.app \"${PLUGINS}\" \"${DIRS}\" )" COMPONENT Runtime)

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
    $<TARGET_FILE:TIFF::TIFF>
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
endif(WIN32)
