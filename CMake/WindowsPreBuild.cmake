if(NOT DEFINED APP_DIR)
  set(APP_DIR "${CPACK_TEMPORARY_INSTALL_DIRECTORY}")
endif()

# remove unneeded dirs from installer
# these directories are a result of the install step for libCZI (in renderlib/io)
file(REMOVE_RECURSE "${APP_DIR}/bin")
file(REMOVE_RECURSE "${APP_DIR}/include")
file(REMOVE_RECURSE "${APP_DIR}/lib")
