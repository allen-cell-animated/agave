target_include_directories(renderlib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
)

target_sources(renderlib PRIVATE
"${CMAKE_CURRENT_SOURCE_DIR}/SerializationConstants.h"
"${CMAKE_CURRENT_SOURCE_DIR}/docWriter.h"
"${CMAKE_CURRENT_SOURCE_DIR}/docWriter.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/docWriterJson.h"
"${CMAKE_CURRENT_SOURCE_DIR}/docWriterJson.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/docWriterYaml.h"
"${CMAKE_CURRENT_SOURCE_DIR}/docWriterYaml.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/docReader.h"
"${CMAKE_CURRENT_SOURCE_DIR}/docReader.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/docReaderJson.h"
"${CMAKE_CURRENT_SOURCE_DIR}/docReaderJson.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/docReaderYaml.h"
"${CMAKE_CURRENT_SOURCE_DIR}/docReaderYaml.cpp"
)


