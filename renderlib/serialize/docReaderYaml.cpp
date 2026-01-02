#include "docReaderYaml.h"

#include "SerializationConstants.h"
#include "core/prty/prtyProperty.hpp"
#include "Logging.h"

#include <fstream>
#include <sstream>
#include <cctype>

docReaderYaml::docReaderYaml()
  : m_current(nullptr)
  , m_nextKey("")
{
  m_root.data = YamlObject();
}

docReaderYaml::~docReaderYaml() {}

bool
docReaderYaml::beginDocument(std::string filePath)
{
  m_filePath = filePath;
  m_root.data = YamlObject();

  if (!parseYaml(filePath)) {
    LOG_ERROR << "Failed to parse YAML file: " << filePath;
    return false;
  }

  m_current = &m_root;

  // Clear the context stack
  while (!m_contextStack.empty()) {
    m_contextStack.pop();
  }

  return true;
}

void
docReaderYaml::endDocument()
{
  if (!m_contextStack.empty()) {
    LOG_ERROR << "endDocument() called with " << m_contextStack.size()
              << " unclosed context(s). Document may be incomplete.";
  }
}

bool
docReaderYaml::beginObject(const std::string& i_name)
{
  if (!m_current) {
    LOG_ERROR << "beginObject() called with null current value";
    return false;
  }

  YamlValue* targetObj = nullptr;

  if (m_contextStack.empty()) {
    // Root level object
    if (m_current->isObject()) {
      YamlObject& obj = m_current->asObject();
      if (obj.find(i_name) != obj.end() && obj[i_name].isObject()) {
        targetObj = &obj[i_name];
      }
    }
  } else {
    const Context& ctx = m_contextStack.top();
    if (ctx.isArray() && ctx.value->isArray()) {
      // Reading object from an array
      YamlArray& arr = ctx.value->asArray();
      if (ctx.arrayIndex < arr.size() && arr[ctx.arrayIndex].isObject()) {
        targetObj = &arr[ctx.arrayIndex];
      }
    } else if (ctx.isObject() && ctx.value->isObject()) {
      // Reading object from an object
      YamlObject& obj = ctx.value->asObject();
      if (obj.find(i_name) != obj.end() && obj[i_name].isObject()) {
        targetObj = &obj[i_name];
      }
    }
  }

  if (!targetObj) {
    LOG_ERROR << "beginObject() - object not found: " << i_name;
    return false;
  }

  pushContext(targetObj, i_name, ContextType::Object);
  return true;
}

void
docReaderYaml::endObject()
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
docReaderYaml::beginList(const std::string& i_name)
{
  if (!m_current) {
    LOG_ERROR << "beginList() called with null current value";
    return false;
  }

  YamlValue* targetArray = nullptr;

  if (m_contextStack.empty()) {
    // Root level array
    if (m_current->isObject()) {
      YamlObject& obj = m_current->asObject();
      if (obj.find(i_name) != obj.end() && obj[i_name].isArray()) {
        targetArray = &obj[i_name];
      }
    }
  } else {
    const Context& ctx = m_contextStack.top();
    if (ctx.isArray() && ctx.value->isArray()) {
      // Reading array from an array
      YamlArray& arr = ctx.value->asArray();
      if (ctx.arrayIndex < arr.size() && arr[ctx.arrayIndex].isArray()) {
        targetArray = &arr[ctx.arrayIndex];
      }
    } else if (ctx.isObject() && ctx.value->isObject()) {
      // Reading array from an object
      YamlObject& obj = ctx.value->asObject();
      if (obj.find(i_name) != obj.end() && obj[i_name].isArray()) {
        targetArray = &obj[i_name];
      }
    }
  }

  if (!targetArray) {
    LOG_ERROR << "beginList() - array not found: " << i_name;
    return false;
  }

  pushContext(targetArray, i_name, ContextType::Array);
  return true;
}

void
docReaderYaml::endList()
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
docReaderYaml::hasKey(const std::string& key)
{
  YamlValue* current = getCurrentValue();
  if (!current || !current->isObject()) {
    return false;
  }
  YamlObject& obj = current->asObject();
  return obj.find(key) != obj.end();
}

std::string
docReaderYaml::peekObjectType()
{
  YamlValue* current = getCurrentValue();
  if (!current || !current->isObject()) {
    return "";
  }

  YamlObject& obj = current->asObject();
  if (obj.find(SerializationConstants::TYPE_KEY) != obj.end() && obj[SerializationConstants::TYPE_KEY].isString()) {
    return obj[SerializationConstants::TYPE_KEY].asString();
  }

  return "";
}

uint32_t
docReaderYaml::peekVersion()
{
  YamlValue* current = getCurrentValue();
  if (!current || !current->isObject()) {
    return 0;
  }

  YamlObject& obj = current->asObject();
  if (obj.find(SerializationConstants::VERSION_KEY) != obj.end() &&
      obj[SerializationConstants::VERSION_KEY].isString()) {
    return stringToInt(obj[SerializationConstants::VERSION_KEY].asString());
  }

  return 0;
}

std::string
docReaderYaml::peekObjectName()
{
  YamlValue* current = getCurrentValue();
  if (!current || !current->isObject()) {
    return "";
  }

  YamlObject& obj = current->asObject();
  if (obj.find(SerializationConstants::NAME_KEY) != obj.end() && obj[SerializationConstants::NAME_KEY].isString()) {
    return obj[SerializationConstants::NAME_KEY].asString();
  }

  return "";
}

bool
docReaderYaml::readPrty(prtyProperty* p)
{
  if (!p) {
    return false;
  }

  // Store the property name for the next read operation
  m_nextKey = p->GetPropertyName();

  // Check if the key exists
  if (!hasKey(m_nextKey.c_str())) {
    LOG_ERROR << "readPrty() - property key not found: " << m_nextKey;
    return false;
  }

  // Let the property read itself
  p->Read(*this);
  return true;
}

bool
docReaderYaml::readBool(const std::string& name)
{
  YamlValue* current = getCurrentValue();
  if (!current) {
    return false;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->isObject()) {
      YamlObject& obj = current->asObject();
      if (obj.find(name) != obj.end() && obj[name].isString()) {
        return stringToBool(obj[name].asString());
      }
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.value->isArray()) {
      YamlArray& arr = ctx.value->asArray();
      if (ctx.arrayIndex < arr.size() && arr[ctx.arrayIndex].isString()) {
        bool value = stringToBool(arr[ctx.arrayIndex].asString());
        ctx.arrayIndex++;
        return value;
      }
    }
  }

  return false;
}

int8_t
docReaderYaml::readInt8(const std::string& name)
{
  YamlValue* current = getCurrentValue();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->isObject()) {
      YamlObject& obj = current->asObject();
      if (obj.find(name) != obj.end() && obj[name].isString()) {
        return static_cast<int8_t>(stringToInt(obj[name].asString()));
      }
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.value->isArray()) {
      YamlArray& arr = ctx.value->asArray();
      if (ctx.arrayIndex < arr.size() && arr[ctx.arrayIndex].isString()) {
        int8_t value = static_cast<int8_t>(stringToInt(arr[ctx.arrayIndex].asString()));
        ctx.arrayIndex++;
        return value;
      }
    }
  }

  return 0;
}

int16_t
docReaderYaml::readInt16(const std::string& name)
{
  YamlValue* current = getCurrentValue();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->isObject()) {
      YamlObject& obj = current->asObject();
      if (obj.find(name) != obj.end() && obj[name].isString()) {
        return static_cast<int16_t>(stringToInt(obj[name].asString()));
      }
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.value->isArray()) {
      YamlArray& arr = ctx.value->asArray();
      if (ctx.arrayIndex < arr.size() && arr[ctx.arrayIndex].isString()) {
        int16_t value = static_cast<int16_t>(stringToInt(arr[ctx.arrayIndex].asString()));
        ctx.arrayIndex++;
        return value;
      }
    }
  }

  return 0;
}

int32_t
docReaderYaml::readInt32(const std::string& name)
{
  YamlValue* current = getCurrentValue();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->isObject()) {
      YamlObject& obj = current->asObject();
      if (obj.find(name) != obj.end() && obj[name].isString()) {
        return stringToInt(obj[name].asString());
      }
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.value->isArray()) {
      YamlArray& arr = ctx.value->asArray();
      if (ctx.arrayIndex < arr.size() && arr[ctx.arrayIndex].isString()) {
        int32_t value = stringToInt(arr[ctx.arrayIndex].asString());
        ctx.arrayIndex++;
        return value;
      }
    }
  }

  return 0;
}

int64_t
docReaderYaml::readInt64(const std::string& name)
{
  YamlValue* current = getCurrentValue();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->isObject()) {
      YamlObject& obj = current->asObject();
      if (obj.find(name) != obj.end() && obj[name].isString()) {
        return stringToInt64(obj[name].asString());
      }
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.value->isArray()) {
      YamlArray& arr = ctx.value->asArray();
      if (ctx.arrayIndex < arr.size() && arr[ctx.arrayIndex].isString()) {
        int64_t value = stringToInt64(arr[ctx.arrayIndex].asString());
        ctx.arrayIndex++;
        return value;
      }
    }
  }

  return 0;
}

uint8_t
docReaderYaml::readUint8(const std::string& name)
{
  YamlValue* current = getCurrentValue();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->isObject()) {
      YamlObject& obj = current->asObject();
      if (obj.find(name) != obj.end() && obj[name].isString()) {
        return static_cast<uint8_t>(stringToUint(obj[name].asString()));
      }
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.value->isArray()) {
      YamlArray& arr = ctx.value->asArray();
      if (ctx.arrayIndex < arr.size() && arr[ctx.arrayIndex].isString()) {
        uint8_t value = static_cast<uint8_t>(stringToUint(arr[ctx.arrayIndex].asString()));
        ctx.arrayIndex++;
        return value;
      }
    }
  }

  return 0;
}

uint16_t
docReaderYaml::readUint16(const std::string& name)
{
  YamlValue* current = getCurrentValue();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->isObject()) {
      YamlObject& obj = current->asObject();
      if (obj.find(name) != obj.end() && obj[name].isString()) {
        return static_cast<uint16_t>(stringToUint(obj[name].asString()));
      }
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.value->isArray()) {
      YamlArray& arr = ctx.value->asArray();
      if (ctx.arrayIndex < arr.size() && arr[ctx.arrayIndex].isString()) {
        uint16_t value = static_cast<uint16_t>(stringToUint(arr[ctx.arrayIndex].asString()));
        ctx.arrayIndex++;
        return value;
      }
    }
  }

  return 0;
}

uint32_t
docReaderYaml::readUint32(const std::string& name)
{
  YamlValue* current = getCurrentValue();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->isObject()) {
      YamlObject& obj = current->asObject();
      if (obj.find(name) != obj.end() && obj[name].isString()) {
        return static_cast<uint32_t>(stringToInt(obj[name].asString()));
      }
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.value->isArray()) {
      YamlArray& arr = ctx.value->asArray();
      if (ctx.arrayIndex < arr.size() && arr[ctx.arrayIndex].isString()) {
        uint32_t value = static_cast<uint32_t>(stringToInt(arr[ctx.arrayIndex].asString()));
        ctx.arrayIndex++;
        return value;
      }
    }
  }

  return 0;
}

uint64_t
docReaderYaml::readUint64(const std::string& name)
{
  YamlValue* current = getCurrentValue();
  if (!current) {
    return 0;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->isObject()) {
      YamlObject& obj = current->asObject();
      if (obj.find(name) != obj.end() && obj[name].isString()) {
        return static_cast<uint64_t>(stringToUint(obj[name].asString()));
      }
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.value->isArray()) {
      YamlArray& arr = ctx.value->asArray();
      if (ctx.arrayIndex < arr.size() && arr[ctx.arrayIndex].isString()) {
        uint64_t value = static_cast<uint64_t>(stringToUint(arr[ctx.arrayIndex].asString()));
        ctx.arrayIndex++;
        return value;
      }
    }
  }

  return 0;
}

float
docReaderYaml::readFloat32(const std::string& name)
{
  YamlValue* current = getCurrentValue();
  if (!current) {
    return 0.0f;
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->isObject()) {
      YamlObject& obj = current->asObject();
      if (obj.find(name) != obj.end() && obj[name].isString()) {
        return stringToFloat(obj[name].asString());
      }
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.value->isArray()) {
      YamlArray& arr = ctx.value->asArray();
      if (ctx.arrayIndex < arr.size() && arr[ctx.arrayIndex].isString()) {
        float value = stringToFloat(arr[ctx.arrayIndex].asString());
        ctx.arrayIndex++;
        return value;
      }
    }
  }

  return 0.0f;
}

std::vector<float>
docReaderYaml::readFloat32Array(const std::string& name)
{
  std::vector<float> result;
  YamlValue* current = getCurrentValue();
  if (!current || !current->isObject()) {
    return result;
  }

  YamlObject& obj = current->asObject();
  if (obj.find(name) != obj.end() && obj[name].isArray()) {
    YamlArray& arr = obj[name].asArray();
    for (const auto& elem : arr) {
      if (elem.isString()) {
        result.push_back(stringToFloat(elem.asString()));
      }
    }
  }

  return result;
}

std::vector<int32_t>
docReaderYaml::readInt32Array(const std::string& name)
{
  std::vector<int32_t> result;
  YamlValue* current = getCurrentValue();
  if (!current || !current->isObject()) {
    return result;
  }

  YamlObject& obj = current->asObject();
  if (obj.find(name) != obj.end() && obj[name].isArray()) {
    YamlArray& arr = obj[name].asArray();
    for (const auto& elem : arr) {
      if (elem.isString()) {
        result.push_back(stringToInt(elem.asString()));
      }
    }
  }

  return result;
}

std::vector<uint32_t>
docReaderYaml::readUint32Array(const std::string& name)
{
  std::vector<uint32_t> result;
  YamlValue* current = getCurrentValue();
  if (!current || !current->isObject()) {
    return result;
  }

  YamlObject& obj = current->asObject();
  if (obj.find(name) != obj.end() && obj[name].isArray()) {
    YamlArray& arr = obj[name].asArray();
    for (const auto& elem : arr) {
      if (elem.isString()) {
        result.push_back(static_cast<uint32_t>(stringToInt(elem.asString())));
      }
    }
  }

  return result;
}

std::string
docReaderYaml::readString(const std::string& name)
{
  YamlValue* current = getCurrentValue();
  if (!current) {
    return "";
  }

  if (m_contextStack.empty() || !m_contextStack.top().isArray()) {
    // Reading from object by key
    if (current->isObject()) {
      YamlObject& obj = current->asObject();
      if (obj.find(name) != obj.end() && obj[name].isString()) {
        return obj[name].asString();
      }
    }
  } else {
    // Reading from array by index
    Context& ctx = m_contextStack.top();
    if (ctx.value->isArray()) {
      YamlArray& arr = ctx.value->asArray();
      if (ctx.arrayIndex < arr.size() && arr[ctx.arrayIndex].isString()) {
        std::string value = arr[ctx.arrayIndex].asString();
        ctx.arrayIndex++;
        return value;
      }
    }
  }

  return "";
}

bool
docReaderYaml::parseYaml(const std::string& filePath)
{
  std::ifstream file(filePath);
  if (!file.is_open()) {
    return false;
  }

  // Skip YAML header if present
  std::string line;
  if (std::getline(file, line) && line != "---") {
    file.seekg(0); // Reset if not a header
  }

  int currentIndent = 0;
  m_root.data = parseObject(file, currentIndent, 0);

  file.close();
  return true;
}

docReaderYaml::YamlObject
docReaderYaml::parseObject(std::ifstream& file, int& currentIndent, int baseIndent)
{
  YamlObject obj;
  std::string line;
  std::streampos lastPos = file.tellg();

  while (std::getline(file, line)) {
    if (line.empty() || line.find_first_not_of(" \t") == std::string::npos) {
      continue; // Skip empty lines
    }

    std::string key;
    int indent = getIndent(line);
    std::string value = parseLine(line, key, indent);

    if (indent < baseIndent) {
      // We've dedented, go back and return
      file.seekg(lastPos);
      currentIndent = indent;
      break;
    }

    if (indent == baseIndent && !key.empty()) {
      if (value.empty()) {
        // Check next line for nested content
        lastPos = file.tellg();
        std::string nextLine;
        if (std::getline(file, nextLine)) {
          int nextIndent = getIndent(nextLine);
          file.seekg(lastPos);

          if (nextIndent > indent) {
            // Parse nested object
            int nestedIndent = 0;
            YamlValue nestedValue;

            if (nextLine.find('-') != std::string::npos && nextLine.find_first_not_of(" \t") == nextLine.find('-')) {
              // It's an array
              nestedValue.data = parseArray(file, nestedIndent, indent + 2);
            } else {
              // It's an object
              nestedValue.data = parseObject(file, nestedIndent, indent + 2);
            }
            obj[key] = nestedValue;
          } else {
            // Empty value
            YamlValue emptyValue;
            emptyValue.data = std::string("");
            obj[key] = emptyValue;
          }
        }
      } else {
        // Simple key-value pair - check if it's an inline array
        YamlValue yamlValue;

        // Check if value is an inline array (starts with [ and ends with ])
        if (!value.empty() && value.front() == '[' && value.back() == ']') {
          // Parse inline array
          YamlArray arr;
          std::string content = value.substr(1, value.length() - 2); // Remove [ and ]

          if (!content.empty()) {
            size_t start = 0;
            size_t end = 0;

            while ((end = content.find(',', start)) != std::string::npos) {
              std::string element = trimString(content.substr(start, end - start));
              if (!element.empty()) {
                YamlValue elemValue;
                elemValue.data = element;
                arr.push_back(elemValue);
              }
              start = end + 1;
            }

            // Add the last element
            std::string element = trimString(content.substr(start));
            if (!element.empty()) {
              YamlValue elemValue;
              elemValue.data = element;
              arr.push_back(elemValue);
            }
          }

          yamlValue.data = arr;
        } else {
          yamlValue.data = value;
        }

        obj[key] = yamlValue;
      }
    }

    lastPos = file.tellg();
  }

  return obj;
}

docReaderYaml::YamlArray
docReaderYaml::parseArray(std::ifstream& file, int& currentIndent, int baseIndent)
{
  YamlArray arr;
  std::string line;
  std::streampos lastPos = file.tellg();

  while (std::getline(file, line)) {
    if (line.empty() || line.find_first_not_of(" \t") == std::string::npos) {
      continue;
    }

    int indent = getIndent(line);

    if (indent < baseIndent) {
      file.seekg(lastPos);
      currentIndent = indent;
      break;
    }

    if (indent == baseIndent) {
      size_t dashPos = line.find('-');
      if (dashPos != std::string::npos && dashPos == line.find_first_not_of(" \t")) {
        std::string value = trimString(line.substr(dashPos + 1));

        if (value.empty()) {
          // Check for nested content
          lastPos = file.tellg();
          std::string nextLine;
          if (std::getline(file, nextLine)) {
            int nextIndent = getIndent(nextLine);
            file.seekg(lastPos);

            if (nextIndent > indent) {
              YamlValue nestedValue;
              int nestedIndent = 0;
              nestedValue.data = parseObject(file, nestedIndent, indent + 2);
              arr.push_back(nestedValue);
            }
          }
        } else {
          YamlValue yamlValue;
          yamlValue.data = value;
          arr.push_back(yamlValue);
        }
      }
    }

    lastPos = file.tellg();
  }

  return arr;
}

std::string
docReaderYaml::parseLine(const std::string& line, std::string& key, int& indent)
{
  indent = getIndent(line);
  size_t colonPos = line.find(':');

  if (colonPos != std::string::npos) {
    key = trimString(line.substr(0, colonPos));
    std::string value = trimString(line.substr(colonPos + 1));
    return value;
  }

  return "";
}

std::string
docReaderYaml::trimString(const std::string& str)
{
  size_t start = str.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  size_t end = str.find_last_not_of(" \t\r\n");
  return str.substr(start, end - start + 1);
}

int
docReaderYaml::getIndent(const std::string& line)
{
  int indent = 0;
  for (char c : line) {
    if (c == ' ') {
      indent++;
    } else if (c == '\t') {
      indent += 2; // Treat tab as 2 spaces
    } else {
      break;
    }
  }
  return indent;
}

bool
docReaderYaml::stringToBool(const std::string& str)
{
  std::string lower = str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  return (lower == "true" || lower == "yes" || lower == "1");
}

int
docReaderYaml::stringToInt(const std::string& str)
{
  try {
    return std::stoi(str);
  } catch (...) {
    return 0;
  }
}

int64_t
docReaderYaml::stringToInt64(const std::string& str)
{
  try {
    return std::stoll(str);
  } catch (...) {
    return 0;
  }
}

uint32_t
docReaderYaml::stringToUint(const std::string& str)
{
  try {
    return static_cast<uint32_t>(std::stoull(str));
  } catch (...) {
    return 0;
  }
}

float
docReaderYaml::stringToFloat(const std::string& str)
{
  try {
    return std::stof(str);
  } catch (...) {
    return 0.0f;
  }
}

void
docReaderYaml::pushContext(YamlValue* val, const std::string& name, ContextType type)
{
  m_contextStack.push(Context(val, name, type));
}

bool
docReaderYaml::popContext(ContextType expectedType)
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

docReaderYaml::YamlValue*
docReaderYaml::getCurrentValue()
{
  if (m_contextStack.empty()) {
    return m_current;
  }
  return m_contextStack.top().value;
}
