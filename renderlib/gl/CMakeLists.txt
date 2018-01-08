target_include_directories(renderlib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
)
target_sources(renderlib PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/Image2D.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Image2D.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Util.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Util.h"
)
