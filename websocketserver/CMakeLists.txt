add_executable(websocketserverapp "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

target_include_directories(websocketserverapp PUBLIC
	"${CMAKE_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${XercesC_INCLUDE_DIRS}	
	${GLM_INCLUDE_DIRS}
)
target_compile_definitions(websocketserverapp PUBLIC
	-DBOOST_ALL_DYN_LINK
)
target_sources(websocketserverapp PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/command.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/commandBuffer.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/cgiparser.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/mainwindow.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/renderer.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/renderrequest.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/streamserver.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/util.cpp"
)

target_link_libraries(websocketserverapp 
	renderlib 
	Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network Qt5::WebSockets Qt5::Xml
	${TIFF_LIBRARIES}
	${Boost_LIBRARIES}
)

if (WIN32)
install(FILES 
	$<TARGET_FILE:Qt5::Widgets>
	$<TARGET_FILE:Qt5::Core>
	$<TARGET_FILE:Qt5::Gui>
	$<TARGET_FILE:Qt5::Xml>
	$<TARGET_FILE:Qt5::Network>
	$<TARGET_FILE:Qt5::WebSockets>
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)
endif (WIN32)
