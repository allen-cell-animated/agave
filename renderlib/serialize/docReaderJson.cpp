#include "docReaderJson.h"

#include "SerializationConstants.h"
#include "core/prty/prtyProperty.hpp"
#include "Logging.h"

#include "json/json.hpp"

#include <fstream>

docReaderJson::docReaderJson()
  : m_root(nullptr)
  , m_current(nullptr)
  , m_nextKey("")
{
}

docReaderJson::~docReaderJson()
{
  if (m_root) {
    delete m_root;
    m_root = nullptr;
  }
}

bool
docReaderJson::beginDocument(std::string filePath)
{
  m_filePath = filePath;
  if (m_root) {
    delete m_root;
  }

  // Read the JSON file
  std::ifstream inFile(m_filePath);
  if (!inFile.is_open()) {
    LOG_ERROR << "Failed to open file for reading: " << m_filePath;
    return false;
  }

  try {
    m_root = new nlohmann::json();
    inFile >> *m_root;
    m_current = m_root;
  } catch (const nlohmann::json::exception& e) {
    LOG_ERROR << "JSON parse error: " << e.what();
    delete m_root;
    m_root = nullptr;
    return false;
  }

  inFile.close();

  // Clear the context stack
  while (!m_contextStack.empty()) {
    m_contextStack.pop();
  }

  return true;
}

void
docReaderJson::endDocument()
{
  if (!m_contextStack.empty()) {
    LOG_ERROR << "endDocument() called with " << m_contextStack.size()
              << " unclosed context(s). Document may be incomplete.";
  }
}

bool
docReaderJson::beginObject(const std::string& i_name)
{
  if (!m_current) {
    LOG_ERROR << "beginObject() called with null current object";
    return false;
  }

  nlohmann::json* targetObj = nullptr;

  if (m_contextStack.empty()) {
    // Root level object
    if (m_current->contains(i_name) && (*m_current)[i_name].is_object()) {
      targetObj = &(*m_current)[i_name];
    } else {
      LOG_ERROR << "beginObject() - key not found or not an object: " << i_name;
      return false;
    }
  } else {
    const Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      // Reading object from an array
      if (ctx.arrayIndex < ctx.jsonObj->size() && (*ctx.jsonObj)[ctx.arrayIndex].is_object()) {
        targetObj = &(*ctx.jsonObj)[ctx.arrayIndex];
      } else {
        LOG_ERROR << "beginObject() - array index out of bounds or not an object";
        return false;
      }
    } else {
      // Reading object from an object
      if (ctx.jsonObj->contains(i_name) && (*ctx.jsonObj)[i_name].is_object()) {
        targetObj = &(*ctx.jsonObj)[i_name];
      } else {
        LOG_ERROR << "beginObject() - key not found or not an object: " << i_name;
        return false;
      }
    }
  }

  pushContext(targetObj, i_name, ContextType::Object);
  return true;
}

void
docReaderJson::endObject()
{
  if (m_contextStack.empty()) {
    LOG_ERROR << "endObject() called with no matching beginObject()";
    return;
  }

  if (!popContext(ContextType::Object)) {
    LOG_ERROR << "endObject() called but current context is not an object";
  }
}

bool
docReaderJson::beginList(const std::string& i_name)
{
  if (!m_current) {
    LOG_ERROR << "beginList() called with null current object";
    return false;
  }

  nlohmann::json* targetArray = nullptr;

  if (m_contextStack.empty()) {
    // Root level array
    if (m_current->contains(i_name) && (*m_current)[i_name].is_array()) {
      targetArray = &(*m_current)[i_name];
    } else {
      LOG_ERROR << "beginList() - key not found or not an array: " << i_name;
      return false;
    }
  } else {
    const Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      // Reading array from an array
      if (ctx.arrayIndex < ctx.jsonObj->size() && (*ctx.jsonObj)[ctx.arrayIndex].is_array()) {
        targetArray = &(*ctx.jsonObj)[ctx.arrayIndex];
      } else {
        LOG_ERROR << "beginList() - array index out of bounds or not an array";
        return false;
      }
    } else {
      // Reading array from an object
      if (ctx.jsonObj->contains(i_name) && (*ctx.jsonObj)[i_name].is_array()) {
        targetArray = &(*ctx.jsonObj)[i_name];
      } else {
        LOG_ERROR << "beginList() - key not found or not an array: " << i_name;
        return false;
      }
    }
  }

  pushContext(targetArray, i_name, ContextType::Array);
  return true;
}

void
docReaderJson::endList()
{
  if (m_contextStack.empty()) {
    LOG_ERROR << "endList() called with no matching beginList()";
    return;
  }

  if (!popContext(ContextType::Array)) {
    LOG_ERROR << "endList() called but current context is not an array";
  }
}

bool
docReaderJson::hasKey(const std::string& key)
{
  nlohmann::json* current = getCurrentObject();
  if (!current || !current->is_object()) {
    return false;
  }
  return current->contains(key);
}

std::string
docReaderJson::peekObjectType()
{
  nlohmann::json* current = getCurrentObject();
  if (!current || !current->is_object()) {
    return "";
  }

  // Look for a "_type" key in the current object
  if (current->contains(SerializationConstants::TYPE_KEY) && (*current)[SerializationConstants::TYPE_KEY].is_string()) {
    return (*current)[SerializationConstants::TYPE_KEY].get<std::string>();
  }

  return "";
}

int
docReaderJson::peekVersion()
{
  nlohmann::json* current = getCurrentObject();
  if (!current || !current->is_object()) {
    return 0;
  }

  // Look for a "_version" key in the current object
  if (current->contains(SerializationConstants::VERSION_KEY) &&
      (*current)[SerializationConstants::VERSION_KEY].is_number_integer()) {
    return (*current)[SerializationConstants::VERSION_KEY].get<int>();
  }

  return 0;
}

void
docReaderJson::readPrty(prtyProperty* p)
{
  if (!p) {
    return;
  }

  // Store the property name for the next read operation
  m_nextKey = p->GetPropertyName();

  // Check if the key exists
  nlohmann::json* current = getCurrentObject();
  if (!current || !current->contains(m_nextKey)) {
    LOG_ERROR << "readPrty() - property key not found: " << m_nextKey;
    return;
  }

  // Let the property read itself
  p->Read(*this);
}

bool
docReaderJson::readBool()
{
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return false;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->contains(m_nextKey) && (*current)[m_nextKey].is_boolean()) {
      return (*current)[m_nextKey].get<bool>();
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.arrayIndex < ctx.jsonObj->size() && (*ctx.jsonObj)[ctx.arrayIndex].is_boolean()) {
      bool value = (*ctx.jsonObj)[ctx.arrayIndex].get<bool>();
      ctx.arrayIndex++;
      return value;
    }
  }

  return false;
}

int8_t
docReaderJson::readInt8()
{
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->contains(m_nextKey) && (*current)[m_nextKey].is_number_integer()) {
      return (*current)[m_nextKey].get<int8_t>();
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.arrayIndex < ctx.jsonObj->size() && (*ctx.jsonObj)[ctx.arrayIndex].is_number_integer()) {
      int8_t value = (*ctx.jsonObj)[ctx.arrayIndex].get<int8_t>();
      ctx.arrayIndex++;
      return value;
    }
  }

  return 0;
}

int32_t
docReaderJson::readInt32()
{
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->contains(m_nextKey) && (*current)[m_nextKey].is_number_integer()) {
      return (*current)[m_nextKey].get<int32_t>();
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.arrayIndex < ctx.jsonObj->size() && (*ctx.jsonObj)[ctx.arrayIndex].is_number_integer()) {
      int32_t value = (*ctx.jsonObj)[ctx.arrayIndex].get<int32_t>();
      ctx.arrayIndex++;
      return value;
    }
  }

  return 0;
}

uint32_t
docReaderJson::readUint32()
{
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->contains(m_nextKey) && (*current)[m_nextKey].is_number_unsigned()) {
      return (*current)[m_nextKey].get<uint32_t>();
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.arrayIndex < ctx.jsonObj->size() && (*ctx.jsonObj)[ctx.arrayIndex].is_number_unsigned()) {
      uint32_t value = (*ctx.jsonObj)[ctx.arrayIndex].get<uint32_t>();
      ctx.arrayIndex++;
      return value;
    }
  }

  return 0;
}

float
docReaderJson::readFloat32()
{
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return 0.0f;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->contains(m_nextKey) && (*current)[m_nextKey].is_number()) {
      return (*current)[m_nextKey].get<float>();
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.arrayIndex < ctx.jsonObj->size() && (*ctx.jsonObj)[ctx.arrayIndex].is_number()) {
      float value = (*ctx.jsonObj)[ctx.arrayIndex].get<float>();
      ctx.arrayIndex++;
      return value;
    }
  }

  return 0.0f;
}

std::vector<float>
docReaderJson::readFloat32Array()
{
  std::vector<float> result;
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return result;
  }

  if (current->contains(m_nextKey) && (*current)[m_nextKey].is_array()) {
    const nlohmann::json& arr = (*current)[m_nextKey];
    for (const auto& elem : arr) {
      if (elem.is_number()) {
        result.push_back(elem.get<float>());
      }
    }
  }

  return result;
}

std::vector<int32_t>
docReaderJson::readInt32Array()
{
  std::vector<int32_t> result;
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return result;
  }

  if (current->contains(m_nextKey) && (*current)[m_nextKey].is_array()) {
    const nlohmann::json& arr = (*current)[m_nextKey];
    for (const auto& elem : arr) {
      if (elem.is_number_integer()) {
        result.push_back(elem.get<int32_t>());
      }
    }
  }

  return result;
}

std::vector<uint32_t>
docReaderJson::readUint32Array()
{
  std::vector<uint32_t> result;
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return result;
  }

  if (current->contains(m_nextKey) && (*current)[m_nextKey].is_array()) {
    const nlohmann::json& arr = (*current)[m_nextKey];
    for (const auto& elem : arr) {
      if (elem.is_number_unsigned()) {
        result.push_back(elem.get<uint32_t>());
      }
    }
  }

  return result;
}

std::string
docReaderJson::readString()
{
  nlohmann::json* current = getCurrentObject();
  if (!current) {
    return "";
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->contains(m_nextKey) && (*current)[m_nextKey].is_string()) {
      return (*current)[m_nextKey].get<std::string>();
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.arrayIndex < ctx.jsonObj->size() && (*ctx.jsonObj)[ctx.arrayIndex].is_string()) {
      std::string value = (*ctx.jsonObj)[ctx.arrayIndex].get<std::string>();
      ctx.arrayIndex++;
      return value;
    }
  }

  return "";
}

void
docReaderJson::pushContext(nlohmann::json* obj, const std::string& name, ContextType type)
{
  m_contextStack.push(Context(obj, name, type));
}

bool
docReaderJson::popContext(ContextType expectedType)
{
  if (m_contextStack.empty()) {
    return false;
  }

  const Context& ctx = m_contextStack.top();
  if (ctx.type != expectedType) {
    return false;
  }

  m_contextStack.pop();
  return true;
}

nlohmann::json*
docReaderJson::getCurrentObject()
{
  if (m_contextStack.empty()) {
    return m_current;
  }
  return m_contextStack.top().jsonObj;
}
