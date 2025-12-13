#pragma once

#include "docReader.h"

#include <fstream>
#include <map>
#include <stack>
#include <string>
#include <variant>
#include <vector>

class docReaderYaml : public docReader
{
public:
  docReaderYaml();
  virtual ~docReaderYaml();

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
  virtual uint32_t peekVersion() override;
  virtual std::string peekObjectName() override;

  // Property reading
  virtual bool readPrty(prtyProperty* p) override;

  // Primitive type reading - all require a name parameter
  virtual bool readBool(const std::string& name) override;
  virtual int8_t readInt8(const std::string& name) override;
  virtual int32_t readInt32(const std::string& name) override;
  virtual uint32_t readUint32(const std::string& name) override;
  virtual float readFloat32(const std::string& name) override;
  virtual std::vector<float> readFloat32Array(const std::string& name) override;
  virtual std::vector<int32_t> readInt32Array(const std::string& name) override;
  virtual std::vector<uint32_t> readUint32Array(const std::string& name) override;
  virtual std::string readString(const std::string& name) override;

private:
  // Simple YAML value types
  struct YamlValue;
  using YamlObject = std::map<std::string, YamlValue>;
  using YamlArray = std::vector<YamlValue>;

  struct YamlValue
  {
    std::variant<std::string, YamlObject, YamlArray> data;

    bool isString() const { return std::holds_alternative<std::string>(data); }
    bool isObject() const { return std::holds_alternative<YamlObject>(data); }
    bool isArray() const { return std::holds_alternative<YamlArray>(data); }

    std::string& asString() { return std::get<std::string>(data); }
    YamlObject& asObject() { return std::get<YamlObject>(data); }
    YamlArray& asArray() { return std::get<YamlArray>(data); }

    const std::string& asString() const { return std::get<std::string>(data); }
    const YamlObject& asObject() const { return std::get<YamlObject>(data); }
    const YamlArray& asArray() const { return std::get<YamlArray>(data); }
  };

  enum class ContextType
  {
    Object,
    Array
  };

  struct Context
  {
    YamlValue* value;
    std::string name;
    ContextType type;
    size_t arrayIndex; // For tracking position in arrays

    Context(YamlValue* val, const std::string& n, ContextType t)
      : value(val)
      , name(n)
      , type(t)
      , arrayIndex(0)
    {
    }

    bool isArray() const { return type == ContextType::Array; }
    bool isObject() const { return type == ContextType::Object; }
  };

  // Parsing helpers
  bool parseYaml(const std::string& filePath);
  YamlValue parseValue(std::ifstream& file, int& currentIndent, int expectedIndent);
  YamlObject parseObject(std::ifstream& file, int& currentIndent, int baseIndent);
  YamlArray parseArray(std::ifstream& file, int& currentIndent, int baseIndent);
  std::string parseLine(const std::string& line, std::string& key, int& indent);
  std::string trimString(const std::string& str);
  int getIndent(const std::string& line);

  // Type conversion helpers
  bool stringToBool(const std::string& str);
  int stringToInt(const std::string& str);
  float stringToFloat(const std::string& str);

  void pushContext(YamlValue* val, const std::string& name, ContextType type);
  bool popContext(ContextType expectedType);
  YamlValue* getCurrentValue();

  YamlValue m_root;
  YamlValue* m_current;
  std::stack<Context> m_contextStack;
  std::string m_nextKey;
  std::string m_filePath;
};
