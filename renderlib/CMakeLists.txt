set (GLAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/glad/include")

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/version.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/version.h"
)

add_library(renderlib "${CMAKE_CURRENT_SOURCE_DIR}/renderlib.cpp")

target_include_directories(renderlib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${ASSIMP_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${XercesC_INCLUDE_DIRS}	
	${GLM_INCLUDE_DIRS}
	${TIFF_INCLUDE_DIR}
	${GLAD_DIR}
	${CUDA_SDK_ROOT_DIR}/common/inc
	${CUDA_TOOLKIT_INCLUDE}
	${OptiX_INCLUDE}
)
target_compile_definitions(renderlib PUBLIC
	-DBOOST_ALL_DYN_LINK
)
target_sources(renderlib PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/AppScene.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/AppScene.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/CCamera.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/command.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/command.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/FileReader.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/FileReader.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/Flags.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Flags.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/HardwareWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/HardwareWidget.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/Histogram.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Histogram.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/ImageXYZC.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ImageXYZC.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/ImageXyzcCuda.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ImageXyzcCuda.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/IRenderWindow.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/IRenderWindow.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/Logging.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Logging.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/OptiXMesh.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/OptiXMesh.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderGL.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderGL.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderGLCuda.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderGLCuda.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderGLOptix.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderGLOptix.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderSettings.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderSettings.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/Status.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Status.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/TexelProperties.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/TexelProperties.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/Timing.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Timing.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/tiny_obj_loader.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/tiny_obj_loader.h"	
)
add_subdirectory(gl)
add_subdirectory(gl/v33)
add_subdirectory(glad/src)
add_subdirectory(glsl/v330)
add_subdirectory(cudarndr)

set(CMAKE_AUTOMOC OFF)
set(CMAKE_AUTOUIC OFF)
add_subdirectory(optixrndr)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

target_link_libraries(renderlib
	Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network Qt5::WebSockets Qt5::Xml 
	renderlib_cuda
	${TIFF_LIBRARIES}
	${optix_LIBRARY}
)
