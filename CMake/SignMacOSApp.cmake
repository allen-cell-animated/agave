# Script for macOS code signing

if(NOT DEFINED APP_DIR)
  set(APP_DIR "${CPACK_TEMPORARY_INSTALL_DIRECTORY}")
endif()

file(GLOB LIBSTOSIGN "${APP_DIR}/agave.app/Contents/Frameworks/*.*")
foreach(LIBTOSIGN ${LIBSTOSIGN})
  execute_process(COMMAND codesign --force --sign - "${LIBTOSIGN}")
endforeach()
execute_process(COMMAND codesign --force --sign - "${APP_DIR}/agave.app")
execute_process(COMMAND xattr -rd com.apple.quarantine "${APP_DIR}/agave.app")