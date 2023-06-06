FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG 3f0283de7a9c43200033da996ff9093be3ac84dc # v3.3.2
)
FetchContent_MakeAvailable(Catch2)

add_executable(agave_test MACOSX_BUNDLE "${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp")
set_target_properties(agave_test PROPERTIES OUTPUT_NAME "agave_test")

target_include_directories(agave_test PUBLIC
  "${CMAKE_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  ${GLM_INCLUDE_DIRS}
)
target_sources(agave_test PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/test_commands.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/test_histogram.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/test_timeLine.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/test_serialize.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/test_version.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/test_volumeDimensions.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../agave_app/commandBuffer.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../agave_app/commandBuffer.h"
)

target_link_libraries(agave_test PRIVATE
  renderlib
  Qt::Widgets Qt::Core Qt::Gui Qt::Network Qt::OpenGL Qt::OpenGLWidgets Qt::WebSockets Qt::Xml
  Catch2WithMain
)

add_custom_command(TARGET agave_test POST_BUILD
  COMMAND agave_test
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test
  COMMENT "Running tests..."
)
