#pragma once

#include <string>

/**
* Check OpenGL status.
*
* Call glGetError() and log the specified message with
* additional details if a problem was encountered.
*
* @param message the message to log on error.
*/
extern void
check_gl(std::string const& message);

