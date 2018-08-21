add_executable(ome2vdb "${CMAKE_CURRENT_SOURCE_DIR}/ome2vdb.cpp")
target_include_directories(ome2vdb PUBLIC
	"${CMAKE_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${Boost_INCLUDE_DIRS}
	${XercesC_INCLUDE_DIRS}	
	${GLM_INCLUDE_DIRS}
	${OPENVDB_INCLUDE_DIR}
)
target_compile_definitions(ome2vdb PUBLIC
	BOOST_ALL_DYN_LINK
)
target_sources(ome2vdb PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/ome2vdb.cpp"
	)
target_link_libraries(ome2vdb 
	renderlib 
	Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network Qt5::WebSockets Qt5::Xml
	${ASSIMP_LIBRARIES}
	${TIFF_LIBRARIES}
	${Boost_LIBRARIES}
	${OPENVDB_LIB}
	${TBB_LIBRARY}
	${HALF_LIB}
)

install(TARGETS ome2vdb
	RUNTIME DESTINATION bin 
)

