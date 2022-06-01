configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/version.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/version.h"
)

add_library(renderlib2 "${CMAKE_CURRENT_SOURCE_DIR}/renderlib2.cpp")
set_property(TARGET renderlib2 PROPERTY AUTOMOC ON)
set_target_properties(renderlib2 PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
target_include_directories(renderlib2 PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${Boost_INCLUDE_DIRS}
	${GLM_INCLUDE_DIRS}
	${GLAD_DIR}
)
target_sources(renderlib2 PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/device.cpp"
)

target_link_libraries(renderlib2
	${CMAKE_DL_LIBS}
	Vulkan::Vulkan

)

IF(WIN32)
	target_link_libraries(renderlib2 glm::glm)
ENDIF(WIN32)

