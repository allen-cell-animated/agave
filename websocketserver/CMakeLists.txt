add_executable(websocketserverapp "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
set_target_properties(websocketserverapp PROPERTIES OUTPUT_NAME "agave-server")
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
	"${CMAKE_CURRENT_SOURCE_DIR}/command.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/commandBuffer.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/commandBuffer.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/cgiparser.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/cgiparser.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/mainwindow.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/mainwindow.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/renderer.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/renderer.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/renderrequest.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/renderrequest.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/streamserver.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/streamserver.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/util.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/util.h"	
)
if (MSVC) 
	target_sources(websocketserverapp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/version.rc")
endif (MSVC)

target_link_libraries(websocketserverapp 
	renderlib 
	Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network Qt5::WebSockets Qt5::Xml
	${ASSIMP_LIBRARIES}
	${TIFF_LIBRARIES}
	${Boost_LIBRARIES}
)

install(TARGETS websocketserverapp
   RUNTIME DESTINATION ${INSTALLDIR} 
)
