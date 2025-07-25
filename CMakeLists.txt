cmake_minimum_required(VERSION 3.24)
include(FetchContent)  # Needed to recognize FetchContent_Declare in renderlib

if(APPLE)
  set(ENV{MACOSX_DEPLOYMENT_TARGET} "10.15")
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version" FORCE)
endif(APPLE)

if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

cmake_policy(SET CMP0091 NEW) # enable new "MSVC runtime library selection" (https://cmake.org/cmake/help/latest/variable/CMAKE_MSVC_RUNTIME_LIBRARY.html)
cmake_policy(SET CMP0135 NEW) # set timestamps to time of extraction for ExternalProject_Add

project(
  agave
  VERSION 1.8.1
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
# or the Qt6<Module>_DIR must be set in the CMake cache to the location of the Qt6WidgetsConfig.cmake
# file. The easiest way to use CMake is to set the CMAKE_PREFIX_PATH environment variable to
# the install prefix of Qt 6.
if(DEFINED ENV{Qt6_DIR})
  list(INSERT CMAKE_PREFIX_PATH 0 $ENV{Qt6_DIR})
endif()

set(AGAVE_QT_VERSION 6.8.3)

if(WIN32)
  set(GUESS_Qt6_DIR C:/Qt/${AGAVE_QT_VERSION}/msvc2022_64 CACHE STRING "Qt6 directory")
elseif(APPLE)
  set(GUESS_Qt6_DIR ~/Qt/${AGAVE_QT_VERSION}/macos)
else()
  set(GUESS_Qt6_DIR ~/Qt/${AGAVE_QT_VERSION}/gcc_64)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif(WIN32)

list(INSERT CMAKE_PREFIX_PATH 0 ${GUESS_Qt6_DIR})

if(APPLE)
  # homebrew
  list(APPEND CMAKE_LIBRARY_PATH /usr/local/lib /usr/local/opt/icu4c/lib)
  list(APPEND CMAKE_INCLUDE_PATH /usr/local/include)

  # set(CMAKE_MACOSX_RPATH TRUE)
  add_compile_options("-I/usr/local/opt/icu4c/include")
  add_link_options("-L/usr/local/opt/icu4c/lib")
endif(APPLE)

find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui OpenGL OpenGLWidgets Network WebSockets Xml Svg REQUIRED)

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

# normally CMAKE_INSTALL_PREFIX is meant to be outside of the build tree
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)
else()
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install/${CMAKE_BUILD_TYPE})
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

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
find_package(Qt6QTiffPlugin 6.8.3 REQUIRED PATHS ${Qt6Gui_DIR})

set(CPACK_PACKAGE_VENDOR "Allen Institute for Cell Science")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AGAVE is a viewer for 4D multichannel microscopy images, using physically based lighting and rendering.")

set(CPACK_PACKAGE_NAME "AGAVE")

install(TARGETS agaveapp
	BUNDLE DESTINATION .
	RUNTIME DESTINATION ${INSTALLDIR}
	LIBRARY DESTINATION ${INSTALLDIR}
	DESTINATION ${INSTALLDIR}
)

if(WIN32)
  install(TARGETS agaveapp
    RUNTIME_DEPENDENCIES
    DIRECTORIES "$<TARGET_FILE_DIR:agaveapp>" "$<TARGET_FILE_DIR:Qt6::Core>"
    PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
    POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
    DESTINATION ${INSTALLDIR}
    CONFIGURATIONS Release
  )
  install(TARGETS agaveapp
    RUNTIME_DEPENDENCIES
    DIRECTORIES "$<TARGET_FILE_DIR:agaveapp>" "$<TARGET_FILE_DIR:Qt6::Core>"
    PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
    POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
    DESTINATION ${INSTALLDIR}
    CONFIGURATIONS Debug
  )
  install(FILES
    ${PROJECT_SOURCE_DIR}/LICENSE.txt
    ${PROJECT_SOURCE_DIR}/HELP.txt
    $<TARGET_FILE:libCZI>
    $<TARGET_FILE:Qt6::Svg>
    $<TARGET_FILE:Qt6::Xml>
    DESTINATION ${INSTALLDIR}
    CONFIGURATIONS Release
  )
  install(FILES
    ${PROJECT_SOURCE_DIR}/LICENSE.txt
    ${PROJECT_SOURCE_DIR}/HELP.txt
    $<TARGET_FILE:libCZI>
    $<TARGET_FILE:Qt6::Svg>
    $<TARGET_FILE:Qt6::Xml>
    DESTINATION ${INSTALLDIR}
    CONFIGURATIONS Debug
  )
  install(FILES
    ${PROJECT_SOURCE_DIR}/renderlib/assets/fonts/Arial.ttf
    ${PROJECT_SOURCE_DIR}/renderlib/assets/fonts/Courier_New.ttf
    DESTINATION ${INSTALLDIR}/assets/fonts/
  )
  install(FILES
    $<TARGET_FILE:Qt6::QWindowsIntegrationPlugin>
    DESTINATION ${INSTALLDIR}/platforms/
  )
  install(FILES
    $<TARGET_FILE:Qt6::QSvgIconPlugin>
    DESTINATION ${INSTALLDIR}/iconengines/
  )
  install(FILES
    $<TARGET_FILE:Qt6::QTiffPlugin>
    DESTINATION ${INSTALLDIR}/imageformats/
  )
  install(FILES
    $<TARGET_FILE:Qt6::QJpegPlugin>
    DESTINATION ${INSTALLDIR}/imageformats/
  )

  set(CPACK_GENERATOR "NSIS")
  set(CPACK_PRE_BUILD_SCRIPTS "${CMAKE_SOURCE_DIR}/CMake/WindowsPreBuild.cmake")
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "AGAVE ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
  set(CPACK_PACKAGE_EXECUTABLES "agave" "AGAVE")
  set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/agave_app/icons/logo.ico")
  set(CPACK_NSIS_MUI_UNIICON "${CMAKE_SOURCE_DIR}/agave_app/icons/logo.ico")
  set(CPACK_NSIS_EXECUTABLES_DIRECTORY "agave-install")
  set(CPACK_NSIS_MENU_LINKS "https://allen-cell-animated.github.io/agave" "AGAVE Docs")
  set(CPACK_NSIS_PACKAGE_NAME "AGAVE ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
  set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "
    WriteRegStr HKCR 'agave' '' 'URL:agave'
    WriteRegStr HKCR 'agave' 'URL Protocol' ''
    WriteRegStr HKCR 'agave\\\\DefaultIcon' '' '\\\"$INSTDIR\\\\agave-install\\\\agave.exe\\\"'
    WriteRegStr HKCR 'agave\\\\shell' '' ''
    WriteRegStr HKCR 'agave\\\\shell\\\\open' '' ''
    WriteRegStr HKCR 'agave\\\\shell\\\\open\\\\command' '' '\\\"$INSTDIR\\\\agave-install\\\\agave.exe\\\" --load \\\"%1\\\"'
  ")
  set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "
    DeleteRegKey HKCR 'agave'
  ")

 include(CPack)

# ###############
elseif(APPLE)
  find_package(Qt6QCocoaIntegrationPlugin 6.8.3 REQUIRED PATHS ${Qt6Gui_DIR})
  find_package(Qt6QMacStylePlugin 6.8.3 REQUIRED PATHS ${Qt6Widgets_DIR})

  # ###############
  set(PACKAGE_OSX_TARGET ${CMAKE_OSX_DEPLOYMENT_TARGET})
  add_definitions(-DPACKAGE_OSX_TARGET)

  # based on code from CMake's QtDialog/CMakeLists.txt
  macro(install_qt6_plugin _qt_plugin_name _qt_plugins_var _prefix)
    get_target_property(_qt_plugin_path "${_qt_plugin_name}" LOCATION)

    if(EXISTS "${_qt_plugin_path}")
      get_filename_component(_qt_plugin_file "${_qt_plugin_path}" NAME)
      get_filename_component(_qt_plugin_type "${_qt_plugin_path}" PATH)
      get_filename_component(_qt_plugin_type "${_qt_plugin_type}" NAME)
      set(_qt_plugin_dest "${_prefix}/Contents/PlugIns/${_qt_plugin_type}")
      install(FILES "${_qt_plugin_path}"
        DESTINATION "${_qt_plugin_dest}")
      set(${_qt_plugins_var}
        "${${_qt_plugins_var}};\$ENV{DEST_DIR}\${CMAKE_INSTALL_PREFIX}/${_qt_plugin_dest}/${_qt_plugin_file}")
    else()
      message(FATAL_ERROR "QT plugin ${_qt_plugin_name} not found")
    endif()
  endmacro()

  install(FILES
    $<TARGET_FILE:Qt6::Widgets>
    $<TARGET_FILE:Qt6::Core>
    $<TARGET_FILE:Qt6::Gui>
    $<TARGET_FILE:Qt6::OpenGL>
    $<TARGET_FILE:Qt6::OpenGLWidgets>
    $<TARGET_FILE:Qt6::Xml>
    $<TARGET_FILE:Qt6::Svg>
    $<TARGET_FILE:Qt6::Network>
    $<TARGET_FILE:Qt6::WebSockets>
    $<TARGET_FILE:TIFF::TIFF>
    DESTINATION agave.app/Contents/Frameworks/
  )

  # calls cmake install command and also adds to the QT_PLUGINS list for use below
  install_qt6_plugin("Qt6::QCocoaIntegrationPlugin" QT_PLUGINS agave.app)
  install_qt6_plugin("Qt6::QTiffPlugin" QT_PLUGINS agave.app)
  install_qt6_plugin("Qt6::QJpegPlugin" QT_PLUGINS agave.app)
  install_qt6_plugin("Qt6::QMacStylePlugin" QT_PLUGINS agave.app)
  install_qt6_plugin("Qt6::QSvgIconPlugin" QT_PLUGINS agave.app)

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
    "[Paths]\nPlugins = PlugIns\n")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
    DESTINATION agave.app/Contents/Resources/)

  install(FILES
    ${PROJECT_SOURCE_DIR}/renderlib/assets/fonts/Arial.ttf
    ${PROJECT_SOURCE_DIR}/renderlib/assets/fonts/Courier_New.ttf
    DESTINATION agave.app/Contents/Resources/assets/fonts/
  )
  install(FILES
    ${PROJECT_SOURCE_DIR}/agave_app/icons/logo.icns
    DESTINATION agave.app/Contents/Resources/
  )

  CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/Info.plist.cmake ${CMAKE_CURRENT_BINARY_DIR}/Info.plist @ONLY)
  set_target_properties(agaveapp PROPERTIES MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_BINARY_DIR}/Info.plist")
	set_target_properties(agaveapp PROPERTIES MACOSX_BUNDLE_ICON_FILE logo.icns)

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
  # list(APPEND DIRS "${Qt6Widgets_DIR}/../..")

  include(InstallRequiredSystemLibraries)

  message(STATUS "APPS: ${APPS}")
  message(STATUS "PLUGINS: ${QT_PLUGINS}")
  message(STATUS "DIRS: ${DIRS}")

  install(CODE "
    include(InstallRequiredSystemLibraries)
    include(BundleUtilities)
    set(BU_CHMOD_BUNDLE_ITEMS TRUE)
    fixup_bundle(\"\${CMAKE_INSTALL_PREFIX}/agave.app\" \"${QT_PLUGINS}\" \"${DIRS}\" )")

  set(CPACK_GENERATOR "DragNDrop")
  set(CPACK_PRE_BUILD_SCRIPTS "${CMAKE_SOURCE_DIR}/CMake/MacOSPreBuild.cmake")
  set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/agave_app/icons/dmg-icon.icns")

  include(CPack)

else() # Linux
  find_package(Qt6QXcbIntegrationPlugin 6.8.3 REQUIRED PATHS ${Qt6Gui_DIR})

  install(FILES
    ${PROJECT_SOURCE_DIR}/LICENSE.txt
    ${PROJECT_SOURCE_DIR}/HELP.txt
    $<TARGET_FILE:Qt6::Widgets>
    $<TARGET_FILE:Qt6::Core>
    $<TARGET_FILE:Qt6::Gui>
    $<TARGET_FILE:Qt6::OpenGL>
    $<TARGET_FILE:Qt6::OpenGLWidgets>
    $<TARGET_FILE:Qt6::Xml>
    $<TARGET_FILE:Qt6::Svg>
    $<TARGET_FILE:Qt6::Network>
    $<TARGET_FILE:Qt6::WebSockets>
    $<TARGET_FILE:TIFF::TIFF>
    DESTINATION ${INSTALLDIR}
  )
  install(FILES
    $<TARGET_FILE:Qt6::QXcbIntegrationPlugin>
    DESTINATION ${INSTALLDIR}/platforms/
  )
  install(FILES
    $<TARGET_FILE:Qt6::QTiffPlugin>
    DESTINATION ${INSTALLDIR}/imageformats/
  )
  install(FILES
    $<TARGET_FILE:Qt6::QJpegPlugin>
    DESTINATION ${INSTALLDIR}/imageformats/
  )
  install(FILES
    $<TARGET_FILE:Qt6::QSvgIconPlugin>
    DESTINATION ${INSTALLDIR}/iconengines/
  )
  install(FILES
    ${PROJECT_SOURCE_DIR}/renderlib/assets/fonts/Arial.ttf
    ${PROJECT_SOURCE_DIR}/renderlib/assets/fonts/Courier_New.ttf
    DESTINATION ${INSTALLDIR}/assets/fonts/
  )
endif(WIN32)
