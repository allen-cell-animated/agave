#version 400 core

in VertexData
{
  vec4 f_colour;
} inData;

out vec4 outputColour;

void main(void) {
  outputColour = inData.f_colour;
}
