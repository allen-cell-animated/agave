# force macOS code signing

if(NOT DEFINED APP_DIR)
  set(APP_DIR "${CPACK_TEMPORARY_INSTALL_DIRECTORY}")
endif()

file(GLOB LIBSTOSIGN "${APP_DIR}/agave.app/Contents/Frameworks/*.*")
foreach(LIBTOSIGN ${LIBSTOSIGN})
  execute_process(COMMAND codesign --force --sign - "${LIBTOSIGN}")
endforeach()
execute_process(COMMAND codesign --force --sign - "${APP_DIR}/agave.app")

# remove unneeded dirs from installer 
# these directories are a result of the install step for libCZI (in renderlib/io)
execute_process(COMMAND rm -rf "${APP_DIR}/include")
execute_process(COMMAND rm -rf "${APP_DIR}/lib")