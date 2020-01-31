add_executable(test MACOSX_BUNDLE "${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp")
set_target_properties(test PROPERTIES OUTPUT_NAME "test")
target_include_directories(test PUBLIC
	"${CMAKE_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${Boost_INCLUDE_DIRS}
	${GLM_INCLUDE_DIRS}
)
target_compile_definitions(test PUBLIC
	BOOST_ALL_DYN_LINK
)
target_sources(test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/test_histogram.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp"
)

target_link_libraries(test 
	renderlib 
	Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network Qt5::WebSockets Qt5::Xml
	${TIFF_LIBRARIES}
	${Boost_LIBRARIES}
	pybind11::embed 
)

add_custom_command(TARGET test POST_BUILD
        COMMAND test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test
        COMMENT "Running tests..."
        )
