#include "GLThickLines.h"

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

static const char* vertex_shader_text =
  R"(
    #version 400 core

    // this is the layout of the stripVerts buffer below
    //layout (location = 0) in vec3 vPos;
    //layout (location = 1) in vec2 vUV;
    //layout (location = 2) in vec4 vCol;
    //layout (location = 3) in uint vCode;

    uniform mat4 projection;
    uniform vec2 resolution;
    uniform int stripVertexOffset;
    uniform int picking;
    uniform float thickness;

// for older gl, make this a texture / use instancing
uniform samplerBuffer stripVerts;
//layout(std430, binding = 0) buffer TVertex
//{
//   vec4 vertex[];
//};

    out vec4 Frag_color;
    out vec2 Frag_UV;

    void main()
    {
    int line_i = (gl_VertexID) / 6;
    int tri_i  = (gl_VertexID) % 6;

    uint vCode = 0;
    vec2 vUV = vec2(0.0, 0.0);
    vec4 vCol = vec4(0.0, 0.0, 0.0, 0.0);

    vec4 va[4];
    // put everything in pixel space so we can apply thickness and
    // compute miters
    for (int i=0; i<4; ++i)
    {
        vec4 vtx;
        vtx.x = texelFetch(stripVerts, (stripVertexOffset + line_i+i)*10 + 0).x;
        vtx.y = texelFetch(stripVerts, (stripVertexOffset + line_i+i)*10 + 1).x;
        vtx.z = texelFetch(stripVerts, (stripVertexOffset + line_i+i)*10 + 2).x;
        vtx.w = 1.0;
        va[i] = projection * vtx;//vertex[line_i+i];
        va[i].xyz /= va[i].w;
        va[i].xy = (va[i].xy + 1.0) * 0.5 * resolution;
    }

    vec2 v_line  = normalize(va[2].xy - va[1].xy);
    vec2 nv_line = vec2(-v_line.y, v_line.x);

    vec4 pos;
    if (tri_i == 0 || tri_i == 1 || tri_i == 3)
    {
        vec2 vtan = va[1].xy - va[0].xy;
        vec2 v_pred  = length(vtan) > 0 ? normalize(vtan) : vtan;
        vec2 v_miter = normalize(nv_line + vec2(-v_pred.y, v_pred.x));

        pos = va[1];
        pos.xy += v_miter * thickness * (tri_i == 1 ? -0.5 : 0.5) / dot(v_miter, nv_line);

        vUV.x = texelFetch(stripVerts, (stripVertexOffset + line_i+1)*10 + 3).x;
        vUV.y = texelFetch(stripVerts, (stripVertexOffset + line_i+1)*10 + 4).x;
        vCol.x = texelFetch(stripVerts, (stripVertexOffset + line_i+1)*10 + 5).x;
        vCol.y = texelFetch(stripVerts, (stripVertexOffset + line_i+1)*10 + 6).x;
        vCol.z = texelFetch(stripVerts, (stripVertexOffset + line_i+1)*10 + 7).x;
        vCol.w = texelFetch(stripVerts, (stripVertexOffset + line_i+1)*10 + 8).x;
        vCode = floatBitsToUint(texelFetch(stripVerts, (stripVertexOffset + line_i+1)*10 + 9).x);
    }
    else
    {
        vec2 vtan = va[3].xy - va[2].xy;
        vec2 v_succ  = length(vtan) > 0 ? normalize(vtan) : vtan;
        vec2 v_miter = normalize(nv_line + vec2(-v_succ.y, v_succ.x));

        pos = va[2];
        pos.xy += v_miter * thickness * (tri_i == 5 ? 0.5 : -0.5) / dot(v_miter, nv_line);

        vUV.x = texelFetch(stripVerts, (stripVertexOffset + line_i+2)*10 + 3).x;
        vUV.y = texelFetch(stripVerts, (stripVertexOffset + line_i+2)*10 + 4).x;

        vCol.x = texelFetch(stripVerts, (stripVertexOffset + line_i+2)*10 + 5).x;
        vCol.y = texelFetch(stripVerts, (stripVertexOffset + line_i+2)*10 + 6).x;
        vCol.z = texelFetch(stripVerts, (stripVertexOffset + line_i+2)*10 + 7).x;
        vCol.w = texelFetch(stripVerts, (stripVertexOffset + line_i+2)*10 + 8).x;

        vCode = floatBitsToUint(texelFetch(stripVerts, (stripVertexOffset + line_i+2)*10 + 9).x);
    }

    // undo the perspective divide and go back to clip space
    pos.xy = pos.xy / resolution * 2.0 - 1.0;
    pos.xyz *= pos.w;
    gl_Position = pos;


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

    }
    )";

// * create an array with the points of the line strip.
// * first and last point define the tangents of the start and end of the line strip,
// so you need to add one pt at start and end
// * if drawing a line loop, then the last point has to be added to the array head,
// and the first point added to the tail
// * store the array of pts in a buffer that the shader can access by indexing (ideally SSBO?)
// * shader doesn't need any vertex coordinates or attributes. It just needs to know the index of the line segment.
// * to get the line segment index we just use the index of the vertex currently being processed (gl_VertexID)
// * to draw a line strip with N points (N-1 segments), we need 6*(N-1) vertices
// * each segment is 2 triangles. glDrawArrays(GL_TRIANGLES, 0, 6*(N-1)) will draw the line strip
// line index = gl_VertexID / 6
// tri index = gl_VertexID % 6
// * Since we are drawing N-1 line segments, but the number of elements in the array is N+2,
// the elements form vertex[line_t] to vertex[line_t+3] can be accessed for each vertex which
// is processed in the vertex shader.
// * vertex[line_t+1] and vertex[line_t+2] are the start and end coordinate of the line segment.
// * vertex[line_t] and vertex[line_t+3] are required to compute the miter.
// thickness is provided in pixels, and so we need to convert it to clip space and use window resolution

std::string vertShader = R"(
#version 460

struct TVertexData
{
//  float v[11];
  vec4 pos;
  vec2 uv;
  vec4 color;
  uint code;
};

// for older gl, make this a texture / use instancing
layout(std430, binding = 0) buffer TVertex
{
   TVertexData vertex[];
};

uniform mat4  u_mvp;
uniform vec2  u_resolution;
uniform float u_thickness;

out vec4 Frag_color;
out vec2 Frag_UV;

void main()
{
    int line_i = gl_VertexID / 6;
    int tri_i  = gl_VertexID % 6;

    vec4 va[4];
    for (int i=0; i<4; ++i)
    {
        va[i] = u_mvp * vertex[line_i+i].pos;
        va[i].xyz /= va[i].w;
        va[i].xy = (va[i].xy + 1.0) * 0.5 * u_resolution;
    }

    vec2 v_line  = normalize(va[2].xy - va[1].xy);
    vec2 nv_line = vec2(-v_line.y, v_line.x);

    vec4 pos;
    if (tri_i == 0 || tri_i == 1 || tri_i == 3)
    {
        vec2 v_pred  = normalize(va[1].xy - va[0].xy);
        vec2 v_miter = normalize(nv_line + vec2(-v_pred.y, v_pred.x));

        pos = va[1];
        pos.xy += v_miter * u_thickness * (tri_i == 1 ? -0.5 : 0.5) / dot(v_miter, nv_line);
    }
    else
    {
        vec2 v_succ  = normalize(va[3].xy - va[2].xy);
        vec2 v_miter = normalize(nv_line + vec2(-v_succ.y, v_succ.x));

        pos = va[2];
        pos.xy += v_miter * u_thickness * (tri_i == 5 ? 0.5 : -0.5) / dot(v_miter, nv_line);
    }

    pos.xy = pos.xy / u_resolution * 2.0 - 1.0;
    pos.xyz *= pos.w;

    Frag_UV = vertex[line_i].uv;
    if (picking == 1) {
      uint vCode = vertex[line_i].code;
      Frag_color = vec4(float(vCode & 0xffu) / 255.0,
                        float((vCode >> 8) & 0xffu) / 255.0,
                        float((vCode >> 16) & 0xffu) / 255.0,
                        1.0);
    }
    else {
      Frag_color = vertex[line_i].color;
    }

    gl_Position = pos;
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

#if 0
// main

GLuint
CreateSSBO(std::vector<glm::vec4>& varray)
{
  GLuint ssbo;
  glGenBuffers(1, &ssbo);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
  glBufferData(GL_SHADER_STORAGE_BUFFER, varray.size() * sizeof(*varray.data()), varray.data(), GL_STATIC_DRAW);
  return ssbo;
}

int
main(void)
{
  if (glfwInit() == 0)
    throw std::runtime_error("error initializing glfw");
  GLFWwindow* window = glfwCreateWindow(800, 600, "GLFW OGL window", nullptr, nullptr);
  if (window == nullptr) {
    glfwTerminate();
    throw std::runtime_error("error initializing window");
  }
  glfwMakeContextCurrent(window);
  if (glewInit() != GLEW_OK)
    throw std::runtime_error("error initializing glew");

  OpenGL::CContext::TDebugLevel debug_level = OpenGL::CContext::TDebugLevel::all;
  OpenGL::CContext context;
  context.Init(debug_level);

  GLuint program = OpenGL::CreateProgram(vertShader, fragShader);
  GLint loc_mvp = glGetUniformLocation(program, "u_mvp");
  GLint loc_res = glGetUniformLocation(program, "u_resolution");
  GLint loc_thi = glGetUniformLocation(program, "u_thickness");

  glUseProgram(program);
  glUniform1f(loc_thi, 20.0);

  GLushort pattern = 0x18ff;
  GLfloat factor = 2.0f;

  std::vector<glm::vec4> varray;
  varray.emplace_back(glm::vec4(0.0f, -1.0f, 0.0f, 1.0f));
  varray.emplace_back(glm::vec4(1.0f, -1.0f, 0.0f, 1.0f));
  for (int u = 0; u <= 90; u += 10) {
    double a = u * M_PI / 180.0;
    double c = cos(a), s = sin(a);
    varray.emplace_back(glm::vec4((float)c, (float)s, 0.0f, 1.0f));
  }
  varray.emplace_back(glm::vec4(-1.0f, 1.0f, 0.0f, 1.0f));
  for (int u = 90; u >= 0; u -= 10) {
    double a = u * M_PI / 180.0;
    double c = cos(a), s = sin(a);
    varray.emplace_back(glm::vec4((float)c - 1.0f, (float)s - 1.0f, 0.0f, 1.0f));
  }
  varray.emplace_back(glm::vec4(1.0f, -1.0f, 0.0f, 1.0f));
  varray.emplace_back(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
  GLuint ssbo = CreateSSBO(varray);

  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
  GLsizei N = (GLsizei)varray.size() - 2;

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

  glm::mat4(project);
  int vpSize[2]{ 0, 0 };
  while (!glfwWindowShouldClose(window)) {
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    if (w != vpSize[0] || h != vpSize[1]) {
      vpSize[0] = w;
      vpSize[1] = h;
      glViewport(0, 0, vpSize[0], vpSize[1]);
      float aspect = (float)w / (float)h;
      project = glm::ortho(-aspect, aspect, -1.0f, 1.0f, -10.0f, 10.0f);
      glUniform2f(loc_res, (float)w, (float)h);
    }

    glClear(GL_COLOR_BUFFER_BIT);

    glm::mat4 modelview1(1.0f);
    modelview1 = glm::translate(modelview1, glm::vec3(-0.6f, 0.0f, 0.0f));
    modelview1 = glm::scale(modelview1, glm::vec3(0.5f, 0.5f, 1.0f));
    glm::mat4 mvp1 = project * modelview1;

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, glm::value_ptr(mvp1));
    glDrawArrays(GL_TRIANGLES, 0, 6 * (N - 1));

    glm::mat4 modelview2(1.0f);
    modelview2 = glm::translate(modelview2, glm::vec3(0.6f, 0.0f, 0.0f));
    modelview2 = glm::scale(modelview2, glm::vec3(0.5f, 0.5f, 1.0f));
    glm::mat4 mvp2 = project * modelview2;

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, glm::value_ptr(mvp2));
    glDrawArrays(GL_TRIANGLES, 0, 6 * (N - 1));

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  glfwTerminate();

  return 0;
}
#endif

GLThickLinesShader::GLThickLinesShader()
  : GLShaderProgram()
{
  utilMakeSimpleProgram(vertex_shader_text, fragment_shader_text);

  m_loc_proj = uniformLocation("projection");
  // m_loc_vpos = attributeLocation("vPos");
  // m_loc_vuv = attributeLocation("vUV");
  // m_loc_vcol = attributeLocation("vCol");
  // m_loc_vcode = attributeLocation("vCode");
  m_loc_thickness = uniformLocation("thickness");
  m_loc_resolution = uniformLocation("resolution");
  m_loc_stripVerts = uniformLocation("stripVerts");
  m_loc_stripVertexOffset = uniformLocation("stripVertexOffset");

  glGenVertexArrays(1, &m_vao);
}
GLThickLinesShader::~GLThickLinesShader()
{
  glDeleteVertexArrays(1, &m_vao);
}

void
GLThickLinesShader::configure(bool display, GLuint textureId)
{
  bind();
  check_gl("bind gesture draw shader");
  // glBindVertexArray(m_vao);

  glUniform1i(uniformLocation("picking"), display ? 0 : 1);
  check_gl("set picking uniform");
  if (display)
    glUniform1i(uniformLocation("Texture"), 0);
  else
    glUniform1i(uniformLocation("Texture"), 1);
  check_gl("set texture uniform");
  // glActiveTexture(GL_TEXTURE0);
  // glBindTexture(GL_TEXTURE_2D, textureId);
  check_gl("bind texture");
}

void
GLThickLinesShader::cleanup()
{
  release();

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_BUFFER, 0);

  // glBindVertexArray(0);
}

void
GLThickLinesShader::setProjMatrix(const glm::mat4& proj)
{
  glUniformMatrix4fv(m_loc_proj, 1, GL_FALSE, glm::value_ptr(proj));
}
