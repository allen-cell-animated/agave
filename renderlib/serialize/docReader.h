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
  virtual bool beginObject(const char* i_name) = 0;
  virtual void endObject() = 0;

  // lists can contain objects or properties.
  virtual bool beginList(const char* i_name) = 0;
  virtual void endList() = 0;

  // Check if a key exists at the current level
  virtual bool hasKey(const char* key) = 0;

  // Peek at object type and version without consuming
  virtual std::string peekObjectType() = 0;
  virtual int peekVersion() = 0;

  // properties will read their name and associated value using the primitive read methods.
  virtual void readPrty(prtyProperty* p) = 0;

  virtual bool readBool() = 0;
  virtual int8_t readInt8() = 0;
  virtual int32_t readInt32() = 0;
  virtual uint32_t readUint32() = 0;
  virtual float readFloat32() = 0;
  virtual std::vector<float> readFloat32Array() = 0;
  virtual std::vector<int32_t> readInt32Array() = 0;
  virtual std::vector<uint32_t> readUint32Array() = 0;
  virtual std::string readString() = 0;

  void readProperties(prtyObject* obj);
};
