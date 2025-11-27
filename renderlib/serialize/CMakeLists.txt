target_include_directories(renderlib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
)

target_sources(renderlib PRIVATE
"${CMAKE_CURRENT_SOURCE_DIR}/docWriter.h"
"${CMAKE_CURRENT_SOURCE_DIR}/docWriterJson.h"
"${CMAKE_CURRENT_SOURCE_DIR}/docWriterJson.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/docWriterYaml.h"
"${CMAKE_CURRENT_SOURCE_DIR}/docWriterYaml.cpp"
)


