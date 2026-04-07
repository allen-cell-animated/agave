target_include_directories(agaveapp PUBLIC
	"${CMAKE_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}"
	"${glm_SOURCE_DIR}"
)
target_sources(agaveapp PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/pyrenderer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pyrenderer.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/RenderInterface.h"
)

