#pragma once

#include <cstdint>
#include <vector>
#include <string>

class prtyProperty;
class prtyObject;

class docReader
{
public:
  docReader() {}
  virtual ~docReader() {}

  // this must be called to start and end the document before any other reading can happen.
  virtual bool beginDocument(std::string filePath) = 0;
  virtual void endDocument() = 0;

  // objects can contain other objects, lists, and properties.
  virtual bool beginObject(const std::string& i_name) = 0;
  virtual void endObject() = 0;

  // lists can contain objects or properties.
  virtual bool beginList(const std::string& i_name) = 0;
  virtual void endList() = 0;

  // Check if a key exists at the current level
  virtual bool hasKey(const std::string& key) = 0;

  // Peek at object type and version without consuming
  virtual std::string peekObjectType() = 0;
  virtual uint32_t peekVersion() = 0;
  virtual std::string peekObjectName() = 0;

  // properties will read their name and associated value using the primitive read methods.
  virtual bool readPrty(prtyProperty* p) = 0;

  // Templated property reader that maps value types to primitive read methods
  template<typename T>
  T readProperty(const std::string& name)
  {
    if constexpr (std::is_same_v<T, bool>) {
      return readBool(name);
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return readInt8(name);
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return readInt32(name);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      return readUint32(name);
    } else if constexpr (std::is_same_v<T, float>) {
      return readFloat32(name);
    } else if constexpr (std::is_same_v<T, std::string>) {
      return readString(name);
    } else if constexpr (std::is_same_v<T, std::vector<float>>) {
      return readFloat32Array(name);
    } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
      return readInt32Array(name);
    } else if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
      return readUint32Array(name);
    } else {
      static_assert(sizeof(T) == 0, "Unsupported type for readProperty");
      return T{};
    }
  }

  // Overload that takes a pointer and assigns to it
  template<typename T>
  void readProperty(const std::string& name, T* value)
  {
    *value = readProperty<T>(name);
  }

  // All primitive read methods now require a name parameter
  virtual bool readBool(const std::string& name) = 0;
  virtual int8_t readInt8(const std::string& name) = 0;
  virtual int32_t readInt32(const std::string& name) = 0;
  virtual uint32_t readUint32(const std::string& name) = 0;
  virtual float readFloat32(const std::string& name) = 0;
  virtual std::vector<float> readFloat32Array(const std::string& name) = 0;
  virtual std::vector<int32_t> readInt32Array(const std::string& name) = 0;
  virtual std::vector<uint32_t> readUint32Array(const std::string& name) = 0;
  virtual std::string readString(const std::string& name) = 0;

  void readProperties(prtyObject* obj);
};
