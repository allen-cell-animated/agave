target_include_directories(renderlib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
)
target_sources(renderlib PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/V33FSQ.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/V33FSQ.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/V33Image3D.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/V33Image3D.h"
)
