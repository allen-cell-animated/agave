#include "ScriptServer.h"

#include "Logging.h"

#include "pybind11/embed.h" // everything needed for embedding
namespace py = pybind11;

ScriptServer::ScriptServer() {}

ScriptServer::~ScriptServer() {}

void
ScriptServer::runScriptFile(const std::string& path)
{
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive
  auto agave_module = py::module::import("agave");
  auto globals = py::globals();

  // use the Python API
  py::print("Hello, World!");
  //  py::eval_file(path, globals);
  try {
    py::exec(R"(
    import agave
    r = agave.renderer()
    r.load_ome_tif("E:\\data\\AICS-12_881_7.ome.tif")
    r.render_iterations(100)
    r.frame_scene()
    r.session("E:\\test.png")
    r.redraw()
  )",
             globals);
  } catch (py::error_already_set& ex) {
    py::print(ex.what());
    LOG_ERROR << ex.what();
  }
}

void
ScriptServer::runScript(const std::string& source)
{}
