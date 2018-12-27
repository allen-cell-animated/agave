add_executable(qtomeapp "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_include_directories(qtomeapp PUBLIC
	"${CMAKE_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${Boost_INCLUDE_DIRS}
	${XercesC_INCLUDE_DIRS}	
	${GLM_INCLUDE_DIRS}
)
target_compile_definitions(qtomeapp PUBLIC
	BOOST_ALL_DYN_LINK
)
target_sources(qtomeapp PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/Aperture.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Aperture.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/ApertureWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ApertureWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/AppearanceDockWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/AppearanceDockWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/AppearanceSettingsWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/AppearanceSettingsWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Camera.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Camera.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/CameraController.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/CameraController.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/CameraDockWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/CameraDockWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/CameraWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/CameraWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Controls.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Controls.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Film.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Film.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/FilmWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/FilmWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Focus.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Focus.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/FocusWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/FocusWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/GLContainer.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/GLContainer.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/GLView3D.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/GLView3D.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/GLWindow.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/GLWindow.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Projection.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Projection.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/ProjectionWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ProjectionWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/RangeWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/RangeWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Section.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Section.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/StatisticsWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/StatisticsWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/StatisticsDockWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/StatisticsDockWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/qtome.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/qtome.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/qtome.qrc"
	"${CMAKE_CURRENT_SOURCE_DIR}/qtome.ui"
	"${CMAKE_CURRENT_SOURCE_DIR}/TransferFunction.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/TransferFunction.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/ViewerState.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ViewerState.h"
	)

if (MSVC) 
	target_sources(qtomeapp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/version.rc")
endif (MSVC)

target_link_libraries(qtomeapp 
	renderlib 
	Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network Qt5::WebSockets Qt5::Xml
	${ASSIMP_LIBRARIES}
	${TIFF_LIBRARIES}
	${Boost_LIBRARIES}
)

install(TARGETS qtomeapp
	RUNTIME DESTINATION bin 
)

