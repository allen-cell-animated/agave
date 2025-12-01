#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "renderlib/serialize/docReader.h"
#include "renderlib/serialize/docReaderJson.h"
#include "renderlib/serialize/docReaderYaml.h"
#include "renderlib/serialize/docWriter.h"
#include "renderlib/serialize/docWriterJson.h"
#include "renderlib/serialize/docWriterYaml.h"
#include "renderlib/serialize/SerializationConstants.h"
#include "renderlib/core/prty/prtyObject.hpp"
#include "renderlib/core/prty/prtyPropertyUIInfo.hpp"
#include "renderlib/core/prty/prtyProperty.hpp"
#include "renderlib/core/prty/prtyInt8.hpp"
#include "renderlib/core/prty/prtyInt32.hpp"
#include "renderlib/core/prty/prtyFloat.hpp"
#include "renderlib/core/prty/prtyText.hpp"
#include "renderlib/core/prty/prtyBoolean.hpp"
#include "renderlib/Logging.h"

#include <filesystem>
#include <fstream>
#include <string>

// Helper function to create a test prtyObject
prtyObject*
createTestObject()
{
  prtyObject* obj = new prtyObject();

  auto intProp = std::make_shared<prtyPropertyUIInfo>(new prtyInt32("testInt", 42), "", "Test Integer");
  auto floatProp = std::make_shared<prtyPropertyUIInfo>(new prtyFloat("testFloat", 3.14f), "", "Test Float");
  auto stringProp = std::make_shared<prtyPropertyUIInfo>(new prtyText("testString", "Hello World"), "", "Test String");
  auto boolProp = std::make_shared<prtyPropertyUIInfo>(new prtyBoolean("testBool", true), "", "Test Boolean");
  auto int8Prop = std::make_shared<prtyPropertyUIInfo>(new prtyInt8("testInt8", 127), "", "Test Int8");

  obj->AddProperty(intProp);
  obj->AddProperty(floatProp);
  obj->AddProperty(stringProp);
  obj->AddProperty(boolProp);
  obj->AddProperty(int8Prop);

  return obj;
}

// Helper function to write a test JSON file
void
createTestJsonFile(const std::string& filePath)
{
  prtyObject* obj = createTestObject();

  docWriterJson writer;
  writer.beginDocument(filePath);
  writer.beginObject("testObject");
  writer.writeProperties(obj);
  writer.endObject();
  writer.endDocument();

  delete obj;
}

// Helper function to write a test YAML file
void
createTestYamlFile(const std::string& filePath)
{
  prtyObject* obj = createTestObject();

  docWriterYaml writer;
  writer.beginDocument(filePath);
  writer.beginObject("testObject");
  writer.writeProperties(obj);
  writer.endObject();
  writer.endDocument();

  delete obj;
}

TEST_CASE("Read prtyObject from JSON", "[serialize][docReader]")
{
  std::string jsonPath = "test_read.json";

  // Create a test file
  createTestJsonFile(jsonPath);
  REQUIRE(std::filesystem::exists(jsonPath));

  // Read the file
  docReaderJson reader;
  REQUIRE(reader.beginDocument(jsonPath));

  // Navigate to the object
  REQUIRE(reader.beginObject("testObject"));

  // Read individual properties
  prtyInt32 testInt("testInt");
  reader.readPrty(&testInt);
  int32_t intValue = reader.readInt32();
  REQUIRE(intValue == 42);

  prtyFloat testFloat("testFloat");
  reader.readPrty(&testFloat);
  float floatValue = reader.readFloat32();
  REQUIRE(floatValue == Catch::Approx(3.14f));

  prtyText testString("testString");
  reader.readPrty(&testString);
  std::string stringValue = reader.readString();
  REQUIRE(stringValue == "Hello World");

  prtyBoolean testBool("testBool");
  reader.readPrty(&testBool);
  bool boolValue = reader.readBool();
  REQUIRE(boolValue == true);

  prtyInt8 testInt8("testInt8");
  reader.readPrty(&testInt8);
  int8_t int8Value = reader.readInt8();
  REQUIRE(int8Value == 127);

  reader.endObject();
  reader.endDocument();

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("Read prtyObject from YAML", "[serialize][docReader]")
{
  std::string yamlPath = "test_read.yaml";

  // Create a test file
  createTestYamlFile(yamlPath);
  REQUIRE(std::filesystem::exists(yamlPath));

  // Read the file
  docReaderYaml reader;
  REQUIRE(reader.beginDocument(yamlPath));

  // Navigate to the object
  REQUIRE(reader.beginObject("testObject"));

  // Read individual properties
  prtyInt32 testInt("testInt");
  reader.readPrty(&testInt);
  int32_t intValue = reader.readInt32();
  REQUIRE(intValue == 42);

  prtyFloat testFloat("testFloat");
  reader.readPrty(&testFloat);
  float floatValue = reader.readFloat32();
  REQUIRE(floatValue == Catch::Approx(3.14f));

  prtyText testString("testString");
  reader.readPrty(&testString);
  std::string stringValue = reader.readString();
  REQUIRE(stringValue == "Hello World");

  prtyBoolean testBool("testBool");
  reader.readPrty(&testBool);
  bool boolValue = reader.readBool();
  REQUIRE(boolValue == true);

  prtyInt8 testInt8("testInt8");
  reader.readPrty(&testInt8);
  int8_t int8Value = reader.readInt8();
  REQUIRE(int8Value == 127);

  reader.endObject();
  reader.endDocument();

  // Cleanup
  std::filesystem::remove(yamlPath);
}

TEST_CASE("Read and write roundtrip JSON", "[serialize][docReader]")
{
  std::string jsonPath = "test_roundtrip.json";

  // Create original object and write
  prtyObject* originalObj = createTestObject();
  docWriterJson writer;
  writer.beginDocument(jsonPath);
  writer.beginObject("testObject");
  writer.writeProperties(originalObj);
  writer.endObject();
  writer.endDocument();

  // Read back
  docReaderJson reader;
  REQUIRE(reader.beginDocument(jsonPath));
  REQUIRE(reader.beginObject("testObject"));

  // Create a new object and read properties into it
  prtyObject* readObj = new prtyObject();

  auto intProp = std::make_shared<prtyPropertyUIInfo>(new prtyInt32("testInt", 0), "", "Test Integer");
  auto floatProp = std::make_shared<prtyPropertyUIInfo>(new prtyFloat("testFloat", 0.0f), "", "Test Float");
  auto stringProp = std::make_shared<prtyPropertyUIInfo>(new prtyText("testString", ""), "", "Test String");
  auto boolProp = std::make_shared<prtyPropertyUIInfo>(new prtyBoolean("testBool", false), "", "Test Boolean");
  auto int8Prop = std::make_shared<prtyPropertyUIInfo>(new prtyInt8("testInt8", 0), "", "Test Int8");

  readObj->AddProperty(intProp);
  readObj->AddProperty(floatProp);
  readObj->AddProperty(stringProp);
  readObj->AddProperty(boolProp);
  readObj->AddProperty(int8Prop);

  // Read properties
  reader.readPrty(intProp->GetProperty(0));
  static_cast<prtyInt32*>(intProp->GetProperty(0))->SetValue(reader.readInt32());

  reader.readPrty(floatProp->GetProperty(0));
  static_cast<prtyFloat*>(floatProp->GetProperty(0))->SetValue(reader.readFloat32());

  reader.readPrty(stringProp->GetProperty(0));
  static_cast<prtyText*>(stringProp->GetProperty(0))->SetValue(reader.readString());

  reader.readPrty(boolProp->GetProperty(0));
  static_cast<prtyBoolean*>(boolProp->GetProperty(0))->SetValue(reader.readBool());

  reader.readPrty(int8Prop->GetProperty(0));
  static_cast<prtyInt8*>(int8Prop->GetProperty(0))->SetValue(reader.readInt8());

  reader.endObject();
  reader.endDocument();

  // Verify values match
  REQUIRE(static_cast<prtyInt32*>(intProp->GetProperty(0))->GetValue() == 42);
  REQUIRE(static_cast<prtyFloat*>(floatProp->GetProperty(0))->GetValue() == Catch::Approx(3.14f));
  REQUIRE(static_cast<prtyText*>(stringProp->GetProperty(0))->GetValue() == "Hello World");
  REQUIRE(static_cast<prtyBoolean*>(boolProp->GetProperty(0))->GetValue() == true);
  REQUIRE(static_cast<prtyInt8*>(int8Prop->GetProperty(0))->GetValue() == 127);

  // Cleanup
  delete originalObj;
  delete readObj;
  std::filesystem::remove(jsonPath);
}

TEST_CASE("Read and write roundtrip YAML", "[serialize][docReader]")
{
  std::string yamlPath = "test_roundtrip.yaml";

  // Create original object and write
  prtyObject* originalObj = createTestObject();
  docWriterYaml writer;
  writer.beginDocument(yamlPath);
  writer.beginObject("testObject");
  writer.writeProperties(originalObj);
  writer.endObject();
  writer.endDocument();

  // Read back
  docReaderYaml reader;
  REQUIRE(reader.beginDocument(yamlPath));
  REQUIRE(reader.beginObject("testObject"));

  // Create a new object and read properties into it
  prtyObject* readObj = new prtyObject();

  auto intProp = std::make_shared<prtyPropertyUIInfo>(new prtyInt32("testInt", 0), "", "Test Integer");
  auto floatProp = std::make_shared<prtyPropertyUIInfo>(new prtyFloat("testFloat", 0.0f), "", "Test Float");
  auto stringProp = std::make_shared<prtyPropertyUIInfo>(new prtyText("testString", ""), "", "Test String");
  auto boolProp = std::make_shared<prtyPropertyUIInfo>(new prtyBoolean("testBool", false), "", "Test Boolean");
  auto int8Prop = std::make_shared<prtyPropertyUIInfo>(new prtyInt8("testInt8", 0), "", "Test Int8");

  readObj->AddProperty(intProp);
  readObj->AddProperty(floatProp);
  readObj->AddProperty(stringProp);
  readObj->AddProperty(boolProp);
  readObj->AddProperty(int8Prop);

  // Read properties
  reader.readPrty(intProp->GetProperty(0));
  static_cast<prtyInt32*>(intProp->GetProperty(0))->SetValue(reader.readInt32());

  reader.readPrty(floatProp->GetProperty(0));
  static_cast<prtyFloat*>(floatProp->GetProperty(0))->SetValue(reader.readFloat32());

  reader.readPrty(stringProp->GetProperty(0));
  static_cast<prtyText*>(stringProp->GetProperty(0))->SetValue(reader.readString());

  reader.readPrty(boolProp->GetProperty(0));
  static_cast<prtyBoolean*>(boolProp->GetProperty(0))->SetValue(reader.readBool());

  reader.readPrty(int8Prop->GetProperty(0));
  static_cast<prtyInt8*>(int8Prop->GetProperty(0))->SetValue(reader.readInt8());

  reader.endObject();
  reader.endDocument();

  // Verify values match
  REQUIRE(static_cast<prtyInt32*>(intProp->GetProperty(0))->GetValue() == 42);
  REQUIRE(static_cast<prtyFloat*>(floatProp->GetProperty(0))->GetValue() == Catch::Approx(3.14f));
  REQUIRE(static_cast<prtyText*>(stringProp->GetProperty(0))->GetValue() == "Hello World");
  REQUIRE(static_cast<prtyBoolean*>(boolProp->GetProperty(0))->GetValue() == true);
  REQUIRE(static_cast<prtyInt8*>(int8Prop->GetProperty(0))->GetValue() == 127);

  // Cleanup
  delete originalObj;
  delete readObj;
  std::filesystem::remove(yamlPath);
}

TEST_CASE("Test peek operations JSON", "[serialize][docReader]")
{
  std::string jsonPath = "test_peek.json";

  // Create a JSON file with type and version
  std::ofstream file(jsonPath);
  file << "{\n";
  file << "  \"" << SerializationConstants::TYPE_KEY << "\": \"Camera\",\n";
  file << "  \"" << SerializationConstants::VERSION_KEY << "\": 2,\n";
  file << "  \"testData\": 123\n";
  file << "}\n";
  file.close();

  // Read and peek
  docReaderJson reader;
  REQUIRE(reader.beginDocument(jsonPath));

  std::string type = reader.peekObjectType();
  REQUIRE(type == "Camera");

  int version = reader.peekVersion();
  REQUIRE(version == 2);

  reader.endDocument();

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("Test peek operations YAML", "[serialize][docReader]")
{
  std::string yamlPath = "test_peek.yaml";

  // Create a YAML file with type and version
  std::ofstream file(yamlPath);
  file << "---\n";
  file << SerializationConstants::TYPE_KEY << ": Camera\n";
  file << SerializationConstants::VERSION_KEY << ": 2\n";
  file << "testData: 123\n";
  file.close();

  // Read and peek
  docReaderYaml reader;
  REQUIRE(reader.beginDocument(yamlPath));

  std::string type = reader.peekObjectType();
  REQUIRE(type == "Camera");

  int version = reader.peekVersion();
  REQUIRE(version == 2);

  reader.endDocument();

  // Cleanup
  std::filesystem::remove(yamlPath);
}

TEST_CASE("Test hasKey operation JSON", "[serialize][docReader]")
{
  std::string jsonPath = "test_haskey.json";

  createTestJsonFile(jsonPath);

  docReaderJson reader;
  REQUIRE(reader.beginDocument(jsonPath));
  REQUIRE(reader.beginObject("testObject"));

  // Check for existing keys
  REQUIRE(reader.hasKey("testInt"));
  REQUIRE(reader.hasKey("testFloat"));
  REQUIRE(reader.hasKey("testString"));
  REQUIRE(reader.hasKey("testBool"));

  // Check for non-existing key
  REQUIRE_FALSE(reader.hasKey("nonExistentKey"));

  reader.endObject();
  reader.endDocument();

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("Test hasKey operation YAML", "[serialize][docReader]")
{
  std::string yamlPath = "test_haskey.yaml";

  createTestYamlFile(yamlPath);

  docReaderYaml reader;
  REQUIRE(reader.beginDocument(yamlPath));
  REQUIRE(reader.beginObject("testObject"));

  // Check for existing keys
  REQUIRE(reader.hasKey("testInt"));
  REQUIRE(reader.hasKey("testFloat"));
  REQUIRE(reader.hasKey("testString"));
  REQUIRE(reader.hasKey("testBool"));

  // Check for non-existing key
  REQUIRE_FALSE(reader.hasKey("nonExistentKey"));

  reader.endObject();
  reader.endDocument();

  // Cleanup
  std::filesystem::remove(yamlPath);
}
