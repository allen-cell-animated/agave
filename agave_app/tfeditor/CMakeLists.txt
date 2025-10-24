target_include_directories(agaveapp PUBLIC
	"${CMAKE_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${GLM_INCLUDE_DIRS}
)

target_sources(agaveapp PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/gradients.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/gradients.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/qcustomplot.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/qcustomplot.h"
)

