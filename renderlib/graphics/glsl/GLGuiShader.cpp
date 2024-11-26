#include "GLGuiShader.h"

static const char* vertex_shader_text =
  R"(
    #version 400 core

    layout (location = 0) in vec3 vPos;
    layout (location = 1) in vec2 vUV;
    layout (location = 2) in vec4 vCol;
    layout (location = 3) in uint vCode;

    uniform mat4 projection;
    uniform int picking;

    out vec4 Frag_color;
    out vec2 Frag_UV;

    void main()
    {
        Frag_UV = vUV;
        if (picking == 1) {
          Frag_color = vec4(float(vCode & 0xffu) / 255.0,
                            float((vCode >> 8) & 0xffu) / 255.0,
                            float((vCode >> 16) & 0xffu) / 255.0,
                            1.0);
        }
        else {
          Frag_color = vCol;
        }
        
        gl_Position = projection * vec4(vPos, 1.0);
    }
    )";

static const char* fragment_shader_text =
  R"(
    #version 400 core
    in vec4 Frag_color;
    in vec2 Frag_UV;
    in vec4 gl_FragCoord;
    uniform int picking;  //< draw for display or for picking? Picking has no texture.
    uniform sampler2D Texture;
    out vec4 outputF;

    const float EPSILON = 0.1;

    void main()
    {
        vec4 result = Frag_color;

        // When drawing selection codes, everything is opaque.
        if (picking == 1) {
          result.w = 1.0;
        }

        // Gesture geometry handshake: any uv value below -64 means
        // no texture lookup. Check VertsCode::k_noTexture
        // (add an epsilon to fix some fp errors. 
        // TODO check to see if highp would have helped)
        if (picking == 0 && Frag_UV.x > -64+EPSILON) {
          result *= texture(Texture, Frag_UV.xy);
        }

        // Gesture geometry handshake: any uv equal to -128 means
        // overlay a checkerboard pattern. Check VertsCode::k_marqueePattern
        if (Frag_UV.s == -128.0) {
            // Create a pixel checkerboard pattern used for marquee
            // selection
            int x = int(gl_FragCoord.x); int y = int(gl_FragCoord.y);
            if (((x+y) & 1) == 0) result = vec4(0,0,0,1);
        }
        outputF = result;
    }
    )";

GLGuiShader::GLGuiShader()
  : GLShaderProgram()
{
  utilMakeSimpleProgram(vertex_shader_text, fragment_shader_text);

  m_loc_proj = uniformLocation("projection");
  m_loc_vpos = attributeLocation("vPos");
  m_loc_vuv = attributeLocation("vUV");
  m_loc_vcol = attributeLocation("vCol");
  m_loc_vcode = attributeLocation("vCode");
}

void
GLGuiShader::configure(bool display, GLuint textureId)
{
  bind();
  check_gl("bind gesture draw shader");

  glEnableVertexAttribArray(m_loc_vpos);
  check_gl("enable vertex attrib array 0");
  glEnableVertexAttribArray(m_loc_vuv);
  check_gl("enable vertex attrib array 1");
  glEnableVertexAttribArray(m_loc_vcol);
  check_gl("enable vertex attrib array 2");
  glEnableVertexAttribArray(m_loc_vcode);
  check_gl("enable vertex attrib array 3");

  glUniform1i(uniformLocation("picking"), display ? 0 : 1);
  check_gl("set picking uniform");
  if (display)
    glUniform1i(uniformLocation("Texture"), 0);
  else
    glUniform1i(uniformLocation("Texture"), 1);
  check_gl("set texture uniform");
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textureId);
  check_gl("bind texture");
}

void
GLGuiShader::cleanup()
{
  release();
  glDisableVertexAttribArray(m_loc_vpos);
  glDisableVertexAttribArray(m_loc_vuv);
  glDisableVertexAttribArray(m_loc_vcol);
  glDisableVertexAttribArray(m_loc_vcode);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, 0);
}
