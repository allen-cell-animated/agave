#version 400 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;

out vec2 vUv;
      
void main()
{
  vUv = uv;
  gl_Position = vec4( position, 1.0 );
}

#version 400 core

uniform sampler2D tTexture0;
in vec2 vUv;
out vec4 out_FragColor;

void main()
{
  out_FragColor = texture(tTexture0, vUv);
}
