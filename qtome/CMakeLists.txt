target_include_directories(qtomeapp PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${Boost_INCLUDE_DIRS}
	${XercesC_INCLUDE_DIRS}	
	${glm_DIR}
)
target_compile_definitions(qtomeapp PUBLIC
	BOOST_ALL_DYN_LINK
)
target_sources(qtomeapp PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/Aperture.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ApertureWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/AppearanceDockWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/AppearanceSettingsWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Camera.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/CameraController.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/CameraDockWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/CameraWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Controls.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Film.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/FilmWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Focus.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/FocusWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/GLContainer.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/GLView2D.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/GLView3D.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/GLWindow.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/NavigationDock2D.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Projection.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ProjectionWidget.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/qtome.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/qtome.qrc"
	"${CMAKE_CURRENT_SOURCE_DIR}/qtome.ui"
	"${CMAKE_CURRENT_SOURCE_DIR}/TransferFunction.cpp"
	)
