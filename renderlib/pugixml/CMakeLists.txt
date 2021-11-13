target_include_directories(renderlib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
)
target_sources(renderlib PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/pugixml.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/pugixml.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/pugiconfig.hpp"
)
