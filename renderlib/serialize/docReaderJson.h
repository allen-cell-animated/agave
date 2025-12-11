#pragma once

#include "docReader.h"

#include "json/json.hpp"

#include <fstream>
#include <stack>
#include <string>

class docReaderJson : public docReader
{
public:
  docReaderJson();
  virtual ~docReaderJson();

  // Document lifecycle
  virtual bool beginDocument(std::string filePath) override;
  virtual void endDocument() override;

  // Object support
  virtual bool beginObject(const std::string& i_name) override;
  virtual void endObject() override;

  // List/array support
  virtual bool beginList(const std::string& i_name) override;
  virtual void endList() override;

  // Key checking
  virtual bool hasKey(const std::string& key) override;

  // Peek operations
  virtual std::string peekObjectType() override;
  virtual int peekVersion() override;

  // Property reading
  virtual void readPrty(prtyProperty* p) override;

  // Primitive type reading
  virtual bool readBool() override;
  virtual int8_t readInt8() override;
  virtual int32_t readInt32() override;
  virtual uint32_t readUint32() override;
  virtual float readFloat32() override;
  virtual std::vector<float> readFloat32Array() override;
  virtual std::vector<int32_t> readInt32Array() override;
  virtual std::vector<uint32_t> readUint32Array() override;
  virtual std::string readString() override;

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
    size_t arrayIndex; // For tracking position in arrays

    Context(nlohmann::json* obj, const std::string& n, ContextType t)
      : jsonObj(obj)
      , name(n)
      , type(t)
      , arrayIndex(0)
    {
    }

    bool isArray() const { return type == ContextType::Array; }
    bool isObject() const { return type == ContextType::Object; }
  };

  void pushContext(nlohmann::json* obj, const std::string& name, ContextType type);
  bool popContext(ContextType expectedType);
  nlohmann::json* getCurrentObject();

  nlohmann::json* m_root;
  nlohmann::json* m_current;
  std::stack<Context> m_contextStack;
  std::string m_nextKey;
  std::string m_filePath;
};
