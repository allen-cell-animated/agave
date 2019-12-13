#include "ScriptServer.h"

#include "pybind11/embed.h" // everything needed for embedding
namespace py = pybind11;

ScriptServer::ScriptServer() {}

ScriptServer::~ScriptServer()
{
}

void
ScriptServer::runScriptFile(const std::string& path)
{
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive
  auto agave_module = py::module::import("agave");
  auto globals = py::globals();

  py::print("Hello, World!"); // use the Python API
  py::eval_file(path, globals);
}

void
ScriptServer::runScript(const std::string& source)
{}
