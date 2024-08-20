if(NOT DEFINED APP_DIR)
  set(APP_DIR "${CPACK_TEMPORARY_INSTALL_DIRECTORY}")
endif()

# remove unneeded dirs from installer
execute_process(COMMAND rd /s /q "${APP_DIR}\\bin")
execute_process(COMMAND rd /s /q "${APP_DIR}\\include")
execute_process(COMMAND rd /s /q "${APP_DIR}\\lib")