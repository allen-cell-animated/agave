#include "glad/glad.h"

#include "GLToneMapShader.h"

#include "Logging.h"

#include <gl/Util.h>
#include <glm.h>

#include <iostream>
#include <sstream>

GLToneMapShader::GLToneMapShader()
  : GLShaderProgram()
  , m_vshader()
  , m_fshader()
{
  m_vshader = new GLShader(GL_VERTEX_SHADER);
  m_vshader->compileSourceCode(R"(
#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;

out vec2 vUv;
      
void main()
{
  vUv = uv;
  gl_Position = vec4( position, 1.0 );
}
	)");

  if (!m_vshader->isCompiled()) {
    LOG_ERROR << "GLToneMapShader: Failed to compile vertex shader\n" << m_vshader->log().toStdString();
  }

  m_fshader = new GLShader(GL_FRAGMENT_SHADER);
  m_fshader->compileSourceCode(R"(
#version 330 core

uniform float gInvExposure;
uniform sampler2D tTexture0;
in vec2 vUv;
out vec4 out_FragColor;
      
vec3 XYZtoRGB(vec3 xyz) {
    return vec3(
        3.240479f*xyz[0] - 1.537150f*xyz[1] - 0.498535f*xyz[2],
        -0.969256f*xyz[0] + 1.875991f*xyz[1] + 0.041556f*xyz[2],
        0.055648f*xyz[0] - 0.204043f*xyz[1] + 1.057311f*xyz[2]
    );
}
      
void main()
{
    vec4 pixelColor = texture(tTexture0, vUv);
    
    //out_FragColor = pixelColor;
    //return;

    pixelColor.rgb = XYZtoRGB(pixelColor.rgb);
      
    pixelColor.rgb = 1.0-exp(-pixelColor.rgb*gInvExposure);
    pixelColor = clamp(pixelColor, 0.0, 1.0);
      
    out_FragColor = pixelColor;
    //out_FragColor = pow(pixelColor, vec4(1.0/2.2));

return;

/*
    /////////////////////
    /////////////////////
    /////////////////////
    /////////////////////
    //// DENOISING FILTER

    vec4 clr00 = pixelColor;
    vec4 rgbsample = clr00;
    // convert XYZ to RGB here.
    rgbsample.rgb = XYZtoRGB(clr00.rgb);
    // tone map!
    rgbsample.rgb = 1.0 - exp(-rgbsample.rgb * gInvExposure);
    rgbsample = clamp(rgbsample, 0.0, 1.0);
    clr00 = rgbsample;

    float fCount = 0.0;
    float SumWeights = 0.0;
    vec3 clr = vec3(0.0, 0.0, 0.0);

    for (int i = -gDenoiseWindowRadius; i <= gDenoiseWindowRadius; i++) {
      for (int j = -gDenoiseWindowRadius; j <= gDenoiseWindowRadius; j++) {
        // sad face...
        if (Y + i < 0)
          continue;
        if (X + j < 0)
          continue;
        if (Y + i >= gFilmHeight)
          continue;
        if (X + j >= gFilmWidth)
          continue;

        vec4 clrIJ = texture(tTexture0, vUv + vec2(i,j));
        rgbsample.rgb = XYZToRGB(clrIJ.rgb);
        // tone map!
        rgbsample.rgb = 1.0 - exp(-rgbsample.rgb * gInvExposure);
        rgbsample = clamp(rgbsample, 0.0, 1.0);

        clrIJ = rgbsample;

        float distanceIJ = vecLen(clr00, clrIJ);

        // gDenoiseNoise = 1/h^2
        //
        float weightIJ = expf(-(distanceIJ * gDenoiseNoise + (float)(i * i + j * j) * gDenoiseInvWindowArea));

        clr.x += clrIJ.x * weightIJ;
        clr.y += clrIJ.y * weightIJ;
        clr.z += clrIJ.z * weightIJ;

        SumWeights += weightIJ;

        fCount += (weightIJ > gDenoiseWeightThreshold) ? gDenoiseInvWindowArea : 0;
      }
    }

    SumWeights = 1.0f / SumWeights;

    clr.rgb *= SumWeights;

    float LerpQ = (fCount > gDenoiseLerpThreshold) ? gDenoiseLerpC : 1.0f - gDenoiseLerpC;

    clr.rgb = mix(clr.rgb, clr00.rgb, LerpQ);
    clr.rgb = clamp(clr.rgb, 0.0, 1.0);

    out_FragColor = vec4(clr.rgb, clr00.a);
*/
}
    )");

  if (!m_fshader->isCompiled()) {
    LOG_ERROR << "GLToneMapShader: Failed to compile fragment shader\n" << m_fshader->log().toStdString();
  }

  addShader(m_vshader);
  addShader(m_fshader);
  link();

  if (!isLinked()) {
    LOG_ERROR << "GLToneMapShader: Failed to link shader program\n" << log().toStdString();
  }

  m_texture = uniformLocation("tTexture0");
  m_InvExposure = uniformLocation("gInvExposure");
}

GLToneMapShader::~GLToneMapShader() {}

void
GLToneMapShader::setShadingUniforms(float invExposure)
{
  glUniform1i(m_texture, 0);
  glUniform1f(m_InvExposure, invExposure);
}
