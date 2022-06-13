add_library(renderlib_wgpu "${CMAKE_CURRENT_SOURCE_DIR}/renderlib_wgpu.cpp")
set_target_properties(renderlib_wgpu PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
target_include_directories(renderlib_wgpu PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${GLM_INCLUDE_DIRS}
)
target_sources(renderlib_wgpu PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/renderlib_wgpu.h"
)

# wgpu-native dep
include(ExternalProject)
ExternalProject_Add(
	wgpu_native
	GIT_REPOSITORY "https://github.com/gfx-rs/wgpu-native.git"
	GIT_TAG "v0.12.0.1"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND cargo build
	COMMAND cargo build --release
	BUILD_IN_SOURCE true
	INSTALL_COMMAND ""
	LOG_BUILD OFF
	LOG_DOWNLOAD OFF   # redirect output to log-file (so that we have less clutter)
	LOG_CONFIGURE OFF  # redirect output to log-file (so that we have less clutter)
)
ExternalProject_Get_Property(wgpu_native BINARY_DIR)

if(MSVC)
    add_definitions(-DWGPU_TARGET=WGPU_TARGET_WINDOWS)
    target_compile_options(renderlib_wgpu PRIVATE /W4)
    set(OS_LIBRARIES "userenv" "ws2_32" "Dwmapi" "dbghelp" "d3dcompiler" "D3D12" "D3D11" "DXGI" "setupapi" "Bcrypt")
elseif(APPLE)
    add_definitions(-DWGPU_TARGET=WGPU_TARGET_MACOS)
    set(OS_LIBRARIES "-framework Cocoa" "-framework CoreVideo" "-framework IOKit" "-framework QuartzCore")
    target_compile_options(renderlib_wgpu PRIVATE -x objective-c++)
else(MSVC)
if(USE_WAYLAND)
    add_definitions(-DWGPU_TARGET=WGPU_TARGET_LINUX_WAYLAND)
else(USE_WAYLAND)
    add_definitions(-DWGPU_TARGET=WGPU_TARGET_LINUX_X11)
endif(USE_WAYLAND)
    target_compile_options(renderlib_wgpu PRIVATE -Wall -Wextra -pedantic)
endif(MSVC)

find_library(WGPU_LIBRARY wgpu_native
    HINTS "${BINARY_DIR}/target/debug"
)

add_dependencies(renderlib_wgpu wgpu_native)
target_link_libraries(renderlib_wgpu
    debug "${BINARY_DIR}/target/debug/libwgpu_native${CMAKE_STATIC_LIBRARY_SUFFIX}"
    optimized "${BINARY_DIR}/target/release/libwgpu_native${CMAKE_STATIC_LIBRARY_SUFFIX}"
)
# get the include dir for the c header file
ExternalProject_Get_Property(wgpu_native SOURCE_DIR)
target_include_directories(renderlib_wgpu PUBLIC ${SOURCE_DIR}/ffi)
target_include_directories(renderlib_wgpu PUBLIC ${SOURCE_DIR}/)
# message(STATUS "wgpu_native: ${SOURCE_DIR}/ffi")


target_link_libraries(renderlib_wgpu
	${CMAKE_DL_LIBS}
	${OS_LIBRARIES}
	${WGPU_LIBRARY}
	spdlog::spdlog_header_only
)
IF(WIN32)
	target_link_libraries(renderlib_wgpu glm::glm)
ENDIF(WIN32)

