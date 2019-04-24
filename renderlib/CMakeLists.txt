set (GLAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/glad/include")

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/version.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/version.h"
)

add_library(renderlib "${CMAKE_CURRENT_SOURCE_DIR}/renderlib.cpp")

target_include_directories(renderlib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${Boost_INCLUDE_DIRS}
	${GLM_INCLUDE_DIRS}
	${TIFF_INCLUDE_DIR}
	${GLAD_DIR}
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
	"${CMAKE_CURRENT_SOURCE_DIR}/Histogram.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Histogram.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/ImageXYZC.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ImageXYZC.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/ImageXyzcGpu.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ImageXyzcGpu.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/IRenderWindow.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/IRenderWindow.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/Logging.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Logging.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderGL.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderGL.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderGLPT.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderGLPT.h"	
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

target_link_libraries(renderlib
	Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network Qt5::WebSockets Qt5::Xml 
	${TIFF_LIBRARIES}
)
