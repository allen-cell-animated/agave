#include "ScriptServer.h"

#include "pyrenderer.h"

#include "pybind11/embed.h" // everything needed for embedding
namespace py = pybind11;

ScriptServer::ScriptServer()
{
  m_renderer = new OffscreenRenderer();

  //  m_ec.m_renderSettings = myVolumeData._renderSettings;
  //  m_ec.m_renderer = this;
  //  m_ec.m_appScene = myVolumeData._scene;
  //  m_ec.m_camera = myVolumeData._camera;
  //  m_ec.m_message = "";
}

ScriptServer::~ScriptServer()
{
  delete m_renderer;
}

void
ScriptServer::runScriptFile(const std::string& path)
{
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive

  py::print("Hello, World!"); // use the Python API
}

void
ScriptServer::runScript(const std::string& source)
{}
