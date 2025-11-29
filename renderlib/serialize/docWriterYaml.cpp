#include "docWriterYaml.h"

#include "core/prty/prtyProperty.hpp"
#include "Logging.h"

#include <fstream>
#include <iomanip>

docWriterYaml::docWriterYaml()
  : m_indentLevel(0)
{
}

docWriterYaml::~docWriterYaml() {}

void
docWriterYaml::beginDocument(std::string filePath)
{
  m_filePath = filePath;
  m_output.str("");
  m_output.clear();
  m_indentLevel = 0;

  // Clear the context stack
  while (!m_contextStack.empty()) {
    m_contextStack.pop();
  }

  // Write YAML header
  m_output << "---\n";
}

void
docWriterYaml::endDocument()
{
  // Validate that all contexts are closed
  if (!m_contextStack.empty()) {
    logError("endDocument() called with " + std::to_string(m_contextStack.size()) +
             " unclosed context(s). Document may be incomplete.");
  }

  // Write the YAML to file
  std::ofstream outFile(m_filePath);
  if (outFile.is_open()) {
    outFile << m_output.str();
    outFile.close();
  }
}

void
docWriterYaml::beginObject(const char* i_name)
{
  if (m_contextStack.empty()) {
    // Root level object
    writeKey(i_name);
    m_output << "\n";
    m_indentLevel++;
  } else {
    Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      // Adding object to an array
      writeIndent();
      m_output << "- ";
      if (i_name && strlen(i_name) > 0) {
        m_output << i_name << ":\n";
      } else {
        m_output << "\n";
      }
      m_indentLevel++;
      ctx.firstItem = false;
    } else {
      // Adding object to an object
      writeIndent();
      writeKey(i_name);
      m_output << "\n";
      m_indentLevel++;
      ctx.firstItem = false;
    }
  }

  pushContext(i_name, ContextType::Object);
}

void
docWriterYaml::endObject()
{
  if (m_contextStack.empty()) {
    logError("endObject() called with no matching beginObject()");
    return;
  }

  if (!popContext(ContextType::Object)) {
    logError("endObject() called but current context is not an object");
  }

  m_indentLevel--;
}

void
docWriterYaml::beginList(const char* i_name)
{
  if (m_contextStack.empty()) {
    // Root level array
    writeKey(i_name);
    m_output << "\n";
    m_indentLevel++;
  } else {
    Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      // Adding array to an array
      writeIndent();
      m_output << "- ";
      if (i_name && strlen(i_name) > 0) {
        m_output << i_name << ":\n";
      } else {
        m_output << "\n";
      }
      m_indentLevel++;
      ctx.firstItem = false;
    } else {
      // Adding array to an object
      writeIndent();
      writeKey(i_name);
      m_output << "\n";
      m_indentLevel++;
      ctx.firstItem = false;
    }
  }

  pushContext(i_name, ContextType::Array);
}

void
docWriterYaml::endList()
{
  if (m_contextStack.empty()) {
    logError("endList() called with no matching beginList()");
    return;
  }

  if (!popContext(ContextType::Array)) {
    logError("endList() called but current context is not an array");
  }

  m_indentLevel--;
}

void
docWriterYaml::writePrty(const prtyProperty* p)
{
  if (!p) {
    return;
  }

  // Store the property name for the next write operation
  m_nextKey = p->GetPropertyName();
  p->Write(*this);
}

size_t
docWriterYaml::writeBool(bool value)
{
  if (m_contextStack.empty()) {
    writeKey(m_nextKey);
    m_output << (value ? "true" : "false") << "\n";
  } else {
    Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      writeIndent();
      m_output << "- " << (value ? "true" : "false") << "\n";
      ctx.firstItem = false;
    } else {
      writeIndent();
      writeKey(m_nextKey);
      m_output << (value ? "true" : "false") << "\n";
      ctx.firstItem = false;
    }
  }

  return sizeof(bool);
}

size_t
docWriterYaml::writeInt8(int8_t value)
{
  if (m_contextStack.empty()) {
    writeKey(m_nextKey);
    m_output << static_cast<int>(value) << "\n";
  } else {
    Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      writeIndent();
      m_output << "- " << static_cast<int>(value) << "\n";
      ctx.firstItem = false;
    } else {
      writeIndent();
      writeKey(m_nextKey);
      m_output << static_cast<int>(value) << "\n";
      ctx.firstItem = false;
    }
  }

  return sizeof(int8_t);
}

size_t
docWriterYaml::writeInt32(int32_t value)
{
  if (m_contextStack.empty()) {
    writeKey(m_nextKey);
    m_output << value << "\n";
  } else {
    Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      writeIndent();
      m_output << "- " << value << "\n";
      ctx.firstItem = false;
    } else {
      writeIndent();
      writeKey(m_nextKey);
      m_output << value << "\n";
      ctx.firstItem = false;
    }
  }

  return sizeof(int32_t);
}

size_t
docWriterYaml::writeUint32(uint32_t value)
{
  if (m_contextStack.empty()) {
    writeKey(m_nextKey);
    m_output << value << "\n";
  } else {
    Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      writeIndent();
      m_output << "- " << value << "\n";
      ctx.firstItem = false;
    } else {
      writeIndent();
      writeKey(m_nextKey);
      m_output << value << "\n";
      ctx.firstItem = false;
    }
  }

  return sizeof(uint32_t);
}

size_t
docWriterYaml::writeFloat32(float value)
{
  if (m_contextStack.empty()) {
    writeKey(m_nextKey);
    m_output << std::setprecision(6) << value << "\n";
  } else {
    Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      writeIndent();
      m_output << "- " << std::setprecision(6) << value << "\n";
      ctx.firstItem = false;
    } else {
      writeIndent();
      writeKey(m_nextKey);
      m_output << std::setprecision(6) << value << "\n";
      ctx.firstItem = false;
    }
  }

  return sizeof(float);
}

size_t
docWriterYaml::writeFloat32Array(const std::vector<float>& value)
{
  return writeFloat32Array(value.size(), value.data());
}

size_t
docWriterYaml::writeFloat32Array(size_t count, const float* values)
{
  if (m_contextStack.empty()) {
    writeKey(m_nextKey);
  } else {
    Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      writeIndent();
      m_output << "- ";
    } else {
      writeIndent();
      writeKey(m_nextKey);
      ctx.firstItem = false;
    }
  }

  // Write as inline array [x, y, z, ...]
  m_output << "[";
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) {
      m_output << ", ";
    }
    m_output << std::setprecision(6) << values[i];
  }
  m_output << "]\n";

  return count * sizeof(float);
}

size_t
docWriterYaml::writeInt32Array(const std::vector<int32_t>& value)
{
  if (m_contextStack.empty()) {
    writeKey(m_nextKey);
  } else {
    Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      writeIndent();
      m_output << "- ";
    } else {
      writeIndent();
      writeKey(m_nextKey);
      ctx.firstItem = false;
    }
  }

  // Write as inline array [x, y, z, ...]
  m_output << "[";
  for (size_t i = 0; i < value.size(); ++i) {
    if (i > 0) {
      m_output << ", ";
    }
    m_output << value[i];
  }
  m_output << "]\n";

  return value.size() * sizeof(int32_t);
}

size_t
docWriterYaml::writeUint32Array(const std::vector<uint32_t>& value)
{
  if (m_contextStack.empty()) {
    writeKey(m_nextKey);
  } else {
    Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      writeIndent();
      m_output << "- ";
    } else {
      writeIndent();
      writeKey(m_nextKey);
      ctx.firstItem = false;
    }
  }

  // Write as inline array [x, y, z, ...]
  m_output << "[";
  for (size_t i = 0; i < value.size(); ++i) {
    if (i > 0) {
      m_output << ", ";
    }
    m_output << value[i];
  }
  m_output << "]\n";

  return value.size() * sizeof(uint32_t);
}

size_t
docWriterYaml::writeString(const std::string& value)
{
  std::string escaped = escapeString(value);

  if (m_contextStack.empty()) {
    writeKey(m_nextKey);
    m_output << escaped << "\n";
  } else {
    Context& ctx = m_contextStack.top();
    if (ctx.isArray()) {
      writeIndent();
      m_output << "- " << escaped << "\n";
      ctx.firstItem = false;
    } else {
      writeIndent();
      writeKey(m_nextKey);
      m_output << escaped << "\n";
      ctx.firstItem = false;
    }
  }

  return value.size();
}

void
docWriterYaml::pushContext(const std::string& name, ContextType type)
{
  m_contextStack.push(Context(name, type));
}

bool
docWriterYaml::popContext(ContextType expectedType)
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
docWriterYaml::writeIndent()
{
  for (int i = 0; i < m_indentLevel; ++i) {
    m_output << "  "; // 2 spaces per indent level
  }
}

void
docWriterYaml::writeKey(const std::string& key)
{
  m_output << key << ": ";
}

std::string
docWriterYaml::escapeString(const std::string& str)
{
  // Check if string needs quoting
  bool needsQuotes = false;

  if (str.empty()) {
    return "\"\"";
  }

  // Check for special characters that require quoting
  for (char c : str) {
    if (c == ':' || c == '#' || c == '\n' || c == '\r' || c == '\t' || c == '"' || c == '\'' || c == '\\' || c == '[' ||
        c == ']' || c == '{' || c == '}' || c == ',' || c == '&' || c == '*' || c == '!' || c == '|' || c == '>' ||
        c == '@' || c == '`') {
      needsQuotes = true;
      break;
    }
  }

  // Check if it starts with special characters
  if (str[0] == '-' || str[0] == '?' || str[0] == ' ') {
    needsQuotes = true;
  }

  if (!needsQuotes) {
    return str;
  }

  // Escape the string and wrap in quotes
  std::string result = "\"";
  for (char c : str) {
    switch (c) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        result += c;
        break;
    }
  }
  result += "\"";

  return result;
}

void
docWriterYaml::logError(const std::string& message)
{
  LOG_ERROR << "docWriterYaml: " << message;
}
