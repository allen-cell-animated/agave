target_include_directories(agaveapp PUBLIC
	"${CMAKE_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${Boost_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
    "pybind11"
)
target_sources(agaveapp PRIVATE
    # "${CMAKE_CURRENT_SOURCE_DIR}/pyagave.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pyrenderer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pyrenderer.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/RenderInterface.h"
    # "${CMAKE_CURRENT_SOURCE_DIR}/ScriptServer.cpp"
    # "${CMAKE_CURRENT_SOURCE_DIR}/ScriptServer.h"
)
add_subdirectory(pybind11)

