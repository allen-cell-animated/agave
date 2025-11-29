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
  virtual void beginObject(const char* i_name) = 0;
  virtual void endObject() = 0;

  // lists can contain objects or properties.
  virtual void beginList(const char* i_name) = 0;
  virtual void endList() = 0;

  // properties will write their name and associated value using the primitive write methods.
  virtual void writePrty(const prtyProperty* p) = 0;

  virtual size_t writeBool(bool) = 0;
  virtual size_t writeInt8(int8_t) = 0;
  virtual size_t writeInt32(int32_t) = 0;
  virtual size_t writeUint32(uint32_t) = 0;
  virtual size_t writeFloat32(float) = 0;
  virtual size_t writeFloat32Array(const std::vector<float>&) = 0;
  virtual size_t writeFloat32Array(size_t count, const float* values) = 0;
  virtual size_t writeInt32Array(const std::vector<int32_t>&) = 0;
  virtual size_t writeUint32Array(const std::vector<uint32_t>&) = 0;
  virtual size_t writeString(const std::string&) = 0;

  void writeProperties(prtyObject* obj);
};
