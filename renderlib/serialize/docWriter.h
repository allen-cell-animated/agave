#pragma once

#include <cstdint>
#include <vector>
#include <string>

class prtyProperty;
class prtyObject;

class docWriter
{
public:
  docWriter() {}
  virtual ~docWriter() {}

  // this must be called to start and end the document before any other writing can happen.
  virtual void beginDocument(std::string filePath) = 0;
  virtual void endDocument() = 0;

  // objects can contain other objects, lists, and properties.
  virtual void beginObject(const std::string& i_name, const std::string& i_objectType, uint32_t version) = 0;
  virtual void endObject() = 0;

  // lists can contain objects or properties.
  virtual void beginList(const std::string& i_name) = 0;
  virtual void endList() = 0;

  // properties will write their name and associated value using the primitive write methods.
  virtual void writePrty(const prtyProperty* p) = 0;

  // Templated property writer that maps value types to primitive write methods
  template<typename T>
  size_t writeProperty(const std::string& name, const T& value)
  {
    if constexpr (std::is_same_v<T, bool>) {
      return writeBool(name, value);
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return writeInt8(name, value);
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return writeInt32(name, value);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      return writeUint32(name, value);
    } else if constexpr (std::is_same_v<T, float>) {
      return writeFloat32(name, value);
    } else if constexpr (std::is_same_v<T, std::string>) {
      return writeString(name, value);
    } else if constexpr (std::is_same_v<T, std::vector<float>>) {
      return writeFloat32Array(name, value);
    } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
      return writeInt32Array(name, value);
    } else if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
      return writeUint32Array(name, value);
    } else {
      static_assert(sizeof(T) == 0, "Unsupported type for writeProperty");
      return 0;
    }
  }

  // Templated integer write method
  template<typename T>
  size_t writeInt(const std::string& name, T value)
  {
    if constexpr (std::is_same_v<T, int8_t>) {
      return writeInt8(name, value);
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return writeInt16(name, value);
    } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) {
      return writeInt32(name, value);
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return writeInt64(name, value);
    } else {
      static_assert(sizeof(T) == 0, "Unsupported signed integer type for writeInt");
      return 0;
    }
  }

  // Templated unsigned integer write method
  template<typename T>
  size_t writeUint(const std::string& name, T value)
  {
    if constexpr (std::is_same_v<T, uint8_t>) {
      return writeUint8(name, value);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
      return writeUint16(name, value);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      return writeUint32(name, value);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return writeUint64(name, value);
    } else {
      static_assert(sizeof(T) == 0, "Unsupported unsigned integer type for writeUint");
      return 0;
    }
  }

  // All primitive write methods now require a name parameter
  virtual size_t writeBool(const std::string& name, bool value) = 0;
  virtual size_t writeInt8(const std::string& name, int8_t value) = 0;
  virtual size_t writeInt16(const std::string& name, int16_t value) = 0;
  virtual size_t writeInt32(const std::string& name, int32_t value) = 0;
  virtual size_t writeInt64(const std::string& name, int64_t value) = 0;
  virtual size_t writeUint8(const std::string& name, uint8_t value) = 0;
  virtual size_t writeUint16(const std::string& name, uint16_t value) = 0;
  virtual size_t writeUint32(const std::string& name, uint32_t value) = 0;
  virtual size_t writeUint64(const std::string& name, uint64_t value) = 0;
  virtual size_t writeFloat32(const std::string& name, float value) = 0;
  virtual size_t writeFloat32Array(const std::string& name, const std::vector<float>& value) = 0;
  virtual size_t writeFloat32Array(const std::string& name, size_t count, const float* values) = 0;
  virtual size_t writeInt32Array(const std::string& name, const std::vector<int32_t>& value) = 0;
  virtual size_t writeUint32Array(const std::string& name, const std::vector<uint32_t>& value) = 0;
  virtual size_t writeString(const std::string& name, const std::string& value) = 0;
};
