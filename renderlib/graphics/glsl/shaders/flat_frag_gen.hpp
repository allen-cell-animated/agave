// Generated C source file containing shader

#include <string>

const std::string flat_frag_chunk_0 = R"(#version 460 core

in VertexData
{
  vec4 f_colour;
} inData;

out vec4 outputColour;

void main(void) {
  outputColour = inData.f_colour;
}
)";

const std::string flat_frag_src = 
    flat_frag_chunk_0;
