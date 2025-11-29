#include "docWriterJson.h"

#include "core/prty/prtyProperty.hpp"
#include "Logging.h"

#include "json/json.hpp"

#include <fstream>

docWriterJson::docWriterJson()
  : m_root(nullptr)
  , m_current(nullptr)
  , m_nextKey("")
{
}

docWriterJson::~docWriterJson()
{
  if (m_root) {
    delete m_root;
    m_root = nullptr;
  }
}

void
docWriterJson::beginDocument(std::string filePath)
{
  m_filePath = filePath;
  if (m_root) {
    delete m_root;
  }
  m_root = new nlohmann::json(nlohmann::json::object());
  m_current = m_root;

  // Clear the context stack
  while (!m_contextStack.empty()) {
    m_contextStack.pop();
  }
}

void
docWriterJson::endDocument()
{
  if (!m_root) {
    return;
  }

  // Validate that all contexts are closed
  if (!m_contextStack.empty()) {
    logError("endDocument() called with " + std::to_string(m_contextStack.size()) +
             " unclosed context(s). Document may be incomplete.");
  }

  // Write the JSON to file
  std::ofstream outFile(m_filePath);
  if (outFile.is_open()) {
    outFile << m_root->dump(2); // Pretty print with 2-space indentation
    outFile.close();
  }
}

void
docWriterJson::beginObject(const char* i_name)
{
  nlohmann::json* newObj = new nlohmann::json(nlohmann::json::object());

  if (m_contextStack.empty()) {
    // Root level object
    (*m_current)[i_name] = *newObj;
    pushContext(&(*m_current)[i_name], i_name, ContextType::Object);
  } else {
    const Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      // Adding object to an array
      ctx.jsonObj->push_back(*newObj);
      pushContext(&ctx.jsonObj->back(), i_name, ContextType::Object);
    } else {
      // Adding object to an object
      (*ctx.jsonObj)[i_name] = *newObj;
      pushContext(&(*ctx.jsonObj)[i_name], i_name, ContextType::Object);
    }
  }

  delete newObj;
}

void
docWriterJson::endObject()
{
  if (m_contextStack.empty()) {
    logError("endObject() called with no matching beginObject()");
    return;
  }

  if (!popContext(ContextType::Object)) {
    logError("endObject() called but current context is not an object");
  }
}

void
docWriterJson::beginList(const char* i_name)
{
  nlohmann::json* newArray = new nlohmann::json(nlohmann::json::array());

  if (m_contextStack.empty()) {
    // Root level array
    (*m_current)[i_name] = *newArray;
    pushContext(&(*m_current)[i_name], i_name, ContextType::Array);
  } else {
    const Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      // Adding array to an array
      ctx.jsonObj->push_back(*newArray);
      pushContext(&ctx.jsonObj->back(), i_name, ContextType::Array);
    } else {
      // Adding array to an object
      (*ctx.jsonObj)[i_name] = *newArray;
      pushContext(&(*ctx.jsonObj)[i_name], i_name, ContextType::Array);
    }
  }

  delete newArray;
}

void
docWriterJson::endList()
{
  if (m_contextStack.empty()) {
    logError("endList() called with no matching beginList()");
    return;
  }

  if (!popContext(ContextType::Array)) {
    logError("endList() called but current context is not an array");
  }
}

void
docWriterJson::writePrty(const prtyProperty* p)
{
  if (!p) {
    return;
  }

  // Store the property name for the next write operation
  m_nextKey = p->GetPropertyName();

  // Note: The actual value writing will be done by the specific write methods
  // (writeInt32, writeFloat32, etc.) which will be called after this method
}

size_t
docWriterJson::writeInt32(int32_t value)
{
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    (*current)[m_nextKey] = value;
  } else {
    current->push_back(value);
  }

  return sizeof(int32_t);
}

size_t
docWriterJson::writeUint32(uint32_t value)
{
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    (*current)[m_nextKey] = value;
  } else {
    current->push_back(value);
  }

  return sizeof(uint32_t);
}

size_t
docWriterJson::writeFloat32(float value)
{
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    (*current)[m_nextKey] = value;
  } else {
    current->push_back(value);
  }

  return sizeof(float);
}

size_t
docWriterJson::writeFloat32Array(const std::vector<float>& value)
{
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return 0;
  }

  nlohmann::json arr = nlohmann::json::array();
  for (float v : value) {
    arr.push_back(v);
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    (*current)[m_nextKey] = arr;
  } else {
    current->push_back(arr);
  }

  return value.size() * sizeof(float);
}

size_t
docWriterJson::writeInt32Array(const std::vector<int32_t>& value)
{
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return 0;
  }

  nlohmann::json arr = nlohmann::json::array();
  for (int32_t v : value) {
    arr.push_back(v);
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    (*current)[m_nextKey] = arr;
  } else {
    current->push_back(arr);
  }

  return value.size() * sizeof(int32_t);
}

size_t
docWriterJson::writeUint32Array(const std::vector<uint32_t>& value)
{
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return 0;
  }

  nlohmann::json arr = nlohmann::json::array();
  for (uint32_t v : value) {
    arr.push_back(v);
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    (*current)[m_nextKey] = arr;
  } else {
    current->push_back(arr);
  }

  return value.size() * sizeof(uint32_t);
}

size_t
docWriterJson::writeString(const std::string& value)
{
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    (*current)[m_nextKey] = value;
  } else {
    current->push_back(value);
  }

  return value.size();
}

void
docWriterJson::pushContext(nlohmann::json* obj, const std::string& name, ContextType type)
{
  m_contextStack.push(Context(obj, name, type));
}

bool
docWriterJson::popContext(ContextType expectedType)
{
  if (m_contextStack.empty()) {
    return false;
  }

  const Context& ctx = m_contextStack.top();
  bool isCorrectType = (ctx.type == expectedType);

  if (!isCorrectType) {
    logError("Mismatched begin/end calls: expected " +
             std::string(expectedType == ContextType::Object ? "Object" : "Array") + " but found " +
             std::string(ctx.type == ContextType::Object ? "Object" : "Array") + " for context '" + ctx.name + "'");
  }

  m_contextStack.pop();
  return isCorrectType;
}

void
docWriterJson::logError(const std::string& message)
{
  LOG_ERROR << "docWriterJson: " << message;
}

nlohmann::json*
docWriterJson::getCurrentObject()
{
  if (m_contextStack.empty()) {
    return m_current;
  }
  return m_contextStack.top().jsonObj;
}
