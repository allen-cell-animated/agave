target_include_directories(renderlib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(renderlib PUBLIC
	BOOST_ALL_DYN_LINK
)
target_sources(renderlib PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/glad.c"
)
