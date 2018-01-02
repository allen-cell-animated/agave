set (GLAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/glad/include")

add_library(renderlib "${CMAKE_CURRENT_SOURCE_DIR}/renderlib.cpp")

target_include_directories(renderlib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${Boost_INCLUDE_DIRS}
	${XercesC_INCLUDE_DIRS}	
	${GLM_INCLUDE_DIRS}
	${TIFF_INCLUDE_DIR}
	${GLAD_DIR}
	${CUDA_SDK_ROOT_DIR}/common/inc
	${CUDA_TOOLKIT_INCLUDE}
)
target_compile_definitions(renderlib PUBLIC
	-DBOOST_ALL_DYN_LINK
)
target_sources(renderlib PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/command.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/FileReader.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Flags.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/HardwareWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ImageXYZC.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ImageXyzcCuda.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/IRenderWindow.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Logging.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderGL.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderGL2d.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/RenderGLCuda.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Scene.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/TexelProperties.cpp"
)
add_subdirectory(gl)
add_subdirectory(gl/v33)
add_subdirectory(glad/src)
add_subdirectory(glsl/v330)
add_subdirectory(cudarndr)

target_link_libraries(renderlib
	Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network Qt5::WebSockets Qt5::Xml 
	renderlib_cuda
	${TIFF_LIBRARIES}
)
