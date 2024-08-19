# remove unneeded dirs from installer 
execute_process(COMMAND rd -r "${APP_DIR}/include")
execute_process(COMMAND rd -r "${APP_DIR}/lib")