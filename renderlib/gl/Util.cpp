#include "Util.h"
#include "glad/glad.h"

#include <iostream>

void check_gl(std::string const& message)
{
GLenum err = GL_NO_ERROR;
while ((err = glGetError()) != GL_NO_ERROR)
    {
    std::cerr << "GL error (" << message << ") :";
    switch(err)
        {
        case GL_INVALID_ENUM:
        std::cerr << "Invalid enum";
        break;
        case GL_INVALID_VALUE:
        std::cerr << "Invalid value";
        break;
        case GL_INVALID_OPERATION:
        std::cerr << "Invalid operation";
        break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
        std::cerr << "Invalid framebuffer operation";
        break;
        case GL_OUT_OF_MEMORY:
        std::cerr << "Out of memory";
        break;
        case GL_STACK_UNDERFLOW:
        std::cerr << "Stack underflow";
        break;
        case GL_STACK_OVERFLOW:
        std::cerr << "Stack overflow";
        break;
        default:
        std::cerr << "Unknown (" << err << ')';
        break;
        }
    std::cerr << std::endl;
    }
}
