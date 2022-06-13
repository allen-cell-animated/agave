add_executable(agaveapp MACOSX_BUNDLE "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
set_target_properties(agaveapp PROPERTIES OUTPUT_NAME "agave")
set_target_properties(agaveapp PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set_property(TARGET agaveapp PROPERTY AUTOMOC ON)
set_property(TARGET agaveapp PROPERTY AUTOUIC ON)
target_include_directories(agaveapp PUBLIC
	"${CMAKE_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${GLM_INCLUDE_DIRS}
)
target_sources(agaveapp PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/agaveGui.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/agaveGui.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/agaveGui.qrc"
	"${CMAKE_CURRENT_SOURCE_DIR}/agaveGui.ui"
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
	"${CMAKE_CURRENT_SOURCE_DIR}/cgiparser.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/cgiparser.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/commandBuffer.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/commandBuffer.h"
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
	"${CMAKE_CURRENT_SOURCE_DIR}/GLView3D.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/GLView3D.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/mainwindow.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/mainwindow.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Projection.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Projection.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/ProjectionWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ProjectionWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/QRenderSettings.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/QRenderSettings.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/RangeWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/RangeWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/renderer.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/renderer.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/renderrequest.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/renderrequest.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Section.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Section.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/StatisticsWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/StatisticsWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/StatisticsDockWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/StatisticsDockWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/streamserver.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/streamserver.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/TimelineDockWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/TimelineDockWidget.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/ViewerState.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ViewerState.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/wgpuView3D.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/wgpuView3D.h"
	)
add_subdirectory(python)

add_subdirectory(tfeditor)

if (MSVC)
	target_sources(agaveapp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/version.rc")
	set_property(
		TARGET agaveapp
		PROPERTY VS_DEBUGGER_WORKING_DIRECTORY
		"${CMAKE_INSTALL_PREFIX}/${INSTALLDIR}"
	)
	set_property(
		TARGET agaveapp
		PROPERTY VS_DEBUGGER_COMMAND
		"${CMAKE_INSTALL_PREFIX}/${INSTALLDIR}/$(TargetFileName)"
	)
endif (MSVC)

target_link_libraries(agaveapp PRIVATE
	renderlib
	renderlib_wgpu
	Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Network Qt5::WebSockets Qt5::Xml
)

install(TARGETS agaveapp
	BUNDLE DESTINATION ${CMAKE_INSTALL_PREFIX}
	RUNTIME DESTINATION ${INSTALLDIR}
)

