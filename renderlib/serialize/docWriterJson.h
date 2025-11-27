#pragma once

#include "docWriter.h"

#include <fstream>
#include <stack>
#include <string>

namespace nlohmann {
class json;
}

class docWriterJson : public docWriter
{
public:
  docWriterJson();
  virtual ~docWriterJson();

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
    nlohmann::json* jsonObj;
    std::string name;
    ContextType type;

    Context(nlohmann::json* obj, const std::string& n, ContextType t)
      : jsonObj(obj)
      , name(n)
      , type(t)
    {
    }

    bool isArray() const { return type == ContextType::Array; }
    bool isObject() const { return type == ContextType::Object; }
  };

  std::string m_filePath;
  nlohmann::json* m_root;
  std::stack<Context> m_contextStack;
  nlohmann::json* m_current;
  std::string m_nextKey;

  void pushContext(nlohmann::json* obj, const std::string& name, ContextType type);
  bool popContext(ContextType expectedType);
  nlohmann::json* getCurrentObject();
  void logError(const std::string& message);
};
