target_include_directories(websocketserver PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${XercesC_INCLUDE_DIRS}	
	${glm_DIR}
)
target_compile_definitions(websocketserver PUBLIC
	-DBOOST_ALL_DYN_LINK
)
target_sources(websocketserver PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/cgiparser.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/mainwindow.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/renderer.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/renderparameters.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/renderrequest.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/streamserver.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/util.cpp"
)
