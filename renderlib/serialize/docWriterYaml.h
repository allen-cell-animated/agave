#pragma once

#include "docWriter.h"

#include <fstream>
#include <stack>
#include <string>
#include <sstream>

class docWriterYaml : public docWriter
{
public:
  docWriterYaml();
  virtual ~docWriterYaml();

  // Document lifecycle
  virtual void beginDocument(std::string filePath) override;
  virtual void endDocument() override;

  // Object support
  virtual void beginObject(const char* i_name) override;
  virtual void endObject() override;

  // List/array support
  virtual void beginList(const char* i_name) override;
  virtual void endList() override;

  // Property writing
  virtual void writePrty(const prtyProperty* p) override;

  // Primitive type writing
  virtual size_t writeInt32(int32_t value) override;
  virtual size_t writeUint32(uint32_t value) override;
  virtual size_t writeFloat32(float value) override;
  virtual size_t writeFloat32Array(const std::vector<float>& value) override;
  virtual size_t writeInt32Array(const std::vector<int32_t>& value) override;
  virtual size_t writeUint32Array(const std::vector<uint32_t>& value) override;
  virtual size_t writeString(const std::string& value) override;

private:
  enum class ContextType
  {
    Object,
    Array
  };

  struct Context
  {
    std::string name;
    ContextType type;
    bool firstItem;

    Context(const std::string& n, ContextType t)
      : name(n)
      , type(t)
      , firstItem(true)
    {
    }

    bool isArray() const { return type == ContextType::Array; }
    bool isObject() const { return type == ContextType::Object; }
  };

  std::string m_filePath;
  std::ostringstream m_output;
  std::stack<Context> m_contextStack;
  std::string m_nextKey;
  int m_indentLevel;

  void pushContext(const std::string& name, ContextType type);
  bool popContext(ContextType expectedType);
  void writeIndent();
  void writeKey(const std::string& key);
  std::string escapeString(const std::string& str);
  void logError(const std::string& message);
};
