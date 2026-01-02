#pragma once

#include "docWriter.h"

#include "json/json.hpp"

#include <fstream>
#include <stack>
#include <string>

class docWriterJson : public docWriter
{
public:
  docWriterJson();
  virtual ~docWriterJson();

  // Document lifecycle
  virtual void beginDocument(std::string filePath) override;
  virtual void endDocument() override;

  // Object support
  virtual void beginObject(const std::string& i_name, const std::string& i_objectType, uint32_t version) override;
  virtual void endObject() override;

  // List/array support
  virtual void beginList(const std::string& i_name) override;
  virtual void endList() override;

  // Property writing
  virtual void writePrty(const prtyProperty* p) override;

  // Primitive type writing - all require a name parameter
  virtual size_t writeBool(const std::string& name, bool value) override;
  virtual size_t writeInt8(const std::string& name, int8_t value) override;
  virtual size_t writeInt16(const std::string& name, int16_t value) override;
  virtual size_t writeInt32(const std::string& name, int32_t value) override;
  virtual size_t writeInt64(const std::string& name, int64_t value) override;
  virtual size_t writeUint8(const std::string& name, uint8_t value) override;
  virtual size_t writeUint16(const std::string& name, uint16_t value) override;
  virtual size_t writeUint32(const std::string& name, uint32_t value) override;
  virtual size_t writeUint64(const std::string& name, uint64_t value) override;
  virtual size_t writeFloat32(const std::string& name, float value) override;
  virtual size_t writeFloat32Array(const std::string& name, const std::vector<float>& value) override;
  virtual size_t writeFloat32Array(const std::string& name, size_t count, const float* values) override;
  virtual size_t writeInt32Array(const std::string& name, const std::vector<int32_t>& value) override;
  virtual size_t writeUint32Array(const std::string& name, const std::vector<uint32_t>& value) override;
  virtual size_t writeString(const std::string& name, const std::string& value) override;

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

  void pushContext(nlohmann::json* obj, const std::string& name, ContextType type);
  bool popContext(ContextType expectedType);
  nlohmann::json* getCurrentObject();
  void logError(const std::string& message);
};
