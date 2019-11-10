target_include_directories(qtomeapp PUBLIC
	"${CMAKE_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${Boost_INCLUDE_DIRS}
	${GLM_INCLUDE_DIRS}
)
target_compile_definitions(qtomeapp PUBLIC
	BOOST_ALL_DYN_LINK
)
target_sources(qtomeapp PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/gradients.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/gradients.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/hoverpoints.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/hoverpoints.h"
)

