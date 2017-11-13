target_include_directories(renderlib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
)
target_sources(renderlib PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/GLBasicVolumeShader.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/V330GLFlatShader2D.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/V330GLImageShader2D.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/V330GLImageShader2DnoLut.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/V330GLLineShader2D.cpp"
)
