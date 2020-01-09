#include "ScriptServer.h"

#include "Logging.h"

#include "pybind11/embed.h" // everything needed for embedding
namespace py = pybind11;

// This is here just as an example of the expected syntax,
// and could be passed to ScriptServer::runScript().
// const std::string testScript = R"(
//     import agave
//     r = agave.renderer()
//     r.load_ome_tif("AICS-12_881_7.ome.tif")
//     r.render_iterations(100)
//     r.frame_scene()
//     r.session("test.png")
//     r.redraw()
//   )";

ScriptServer::ScriptServer() {}

ScriptServer::~ScriptServer() {}

void
ScriptServer::runScriptFile(const std::string& path)
{
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive
  auto agave_module = py::module::import("agave");
  auto globals = py::globals();

  // use the Python API
  LOG_INFO << "Begin Python script interpreter";
  try {
    py::eval_file(path, globals);
    // py::exec(testScript, globals);
  } catch (py::error_already_set& ex) {
    py::print(ex.what());
    LOG_ERROR << ex.what();
  }
  LOG_INFO << "End Python script interpreter";
}

void
ScriptServer::runScript(const std::string& source)
{
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive
  auto agave_module = py::module::import("agave");
  auto globals = py::globals();

  // use the Python API
  LOG_INFO << "Begin Python script interpreter";
  try {
    py::exec(source, globals);
  } catch (py::error_already_set& ex) {
    py::print(ex.what());
    LOG_ERROR << ex.what();
  }
  LOG_INFO << "End Python script interpreter";
}
