target_include_directories(agaveapp PUBLIC
	"${CMAKE_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}"
)
target_sources(agaveapp PRIVATE
"${CMAKE_CURRENT_SOURCE_DIR}/Controls.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/Controls.h"
"${CMAKE_CURRENT_SOURCE_DIR}/RangeWidget.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/RangeWidget.h"
"${CMAKE_CURRENT_SOURCE_DIR}/Section.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/Section.h"
)