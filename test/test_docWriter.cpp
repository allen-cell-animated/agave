#include <catch2/catch_test_macros.hpp>

#include "renderlib/serialize/docWriter.h"
#include "renderlib/serialize/docWriterJson.h"
#include "renderlib/serialize/docWriterYaml.h"
#include "renderlib/AppearanceObject.hpp"
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
#include <iostream>

// Helper function to write a prtyObject to a docWriter
void
writePrtyObject(docWriter& writer, prtyObject* obj, const std::string& name)
{
  if (!obj) {
    return;
  }

  writer.beginObject(name.c_str());

  const PropertyUIIList& propList = obj->GetList();
  std::cout << "Property list size: " << propList.size() << std::endl;

  for (const auto& propUIInfo : propList) {
    int numProps = propUIInfo->GetNumberOfProperties();
    std::cout << "Number of properties in UIInfo: " << numProps << std::endl;

    prtyProperty* prop = propUIInfo->GetProperty(0);
    if (!prop) {
      std::cout << "Property is null!" << std::endl;
      continue;
    }

    const char* type = prop->GetType();
    std::string propName = prop->GetPropertyName();
    std::cout << "Writing property: " << propName << " of type: " << type << std::endl;

    // Set up the property name for writing
    writer.writePrty(prop);

    if (strcmp(type, "Int8") == 0) {
      auto* p = static_cast<prtyInt8*>(prop);
      writer.writeInt32(p->GetValue());
    } else if (strcmp(type, "Int32") == 0) {
      auto* p = static_cast<prtyInt32*>(prop);
      writer.writeInt32(p->GetValue());
    } else if (strcmp(type, "Float") == 0) {
      auto* p = static_cast<prtyFloat*>(prop);
      writer.writeFloat32(p->GetValue());
    } else if (strcmp(type, "Text") == 0) {
      auto* p = static_cast<prtyText*>(prop);
      writer.writeString(p->GetValue());
    } else if (strcmp(type, "Boolean") == 0) {
      auto* p = static_cast<prtyBoolean*>(prop);
      writer.writeInt32(p->GetValue() ? 1 : 0);
    }
  }

  writer.endObject();
}

TEST_CASE("Serialize prtyObject to JSON", "[serialize][docWriter]")
{
  // Create a simple prtyObject with some properties
  prtyObject obj;

  auto intProp = std::make_shared<prtyPropertyUIInfo>(new prtyInt32("testInt", 42), "", "Test Integer");
  auto floatProp = std::make_shared<prtyPropertyUIInfo>(new prtyFloat("testFloat", 3.14f), "", "Test Float");
  auto stringProp = std::make_shared<prtyPropertyUIInfo>(new prtyText("testString", "Hello World"), "", "Test String");
  auto boolProp = std::make_shared<prtyPropertyUIInfo>(new prtyBoolean("testBool", true), "", "Test Boolean");

  obj.AddProperty(intProp);
  obj.AddProperty(floatProp);
  obj.AddProperty(stringProp);
  obj.AddProperty(boolProp);

  // Write to JSON
  docWriterJson jsonWriter;
  std::string jsonPath = "test_output.json";

  jsonWriter.beginDocument(jsonPath);
  writePrtyObject(jsonWriter, &obj, "testObject");
  jsonWriter.endDocument();

  // Verify the file was created
  REQUIRE(std::filesystem::exists(jsonPath));

  // Read and verify contents
  std::ifstream file(jsonPath);
  REQUIRE(file.is_open());

  std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  // Check that the JSON contains expected values
  REQUIRE(content.find("testObject") != std::string::npos);
  REQUIRE(content.find("testInt") != std::string::npos);
  REQUIRE(content.find("42") != std::string::npos);
  REQUIRE(content.find("testFloat") != std::string::npos);
  REQUIRE(content.find("3.14") != std::string::npos);
  REQUIRE(content.find("testString") != std::string::npos);
  REQUIRE(content.find("Hello World") != std::string::npos);
  REQUIRE(content.find("testBool") != std::string::npos);

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("Serialize prtyObject to YAML", "[serialize][docWriter]")
{
  // Create a simple prtyObject with some properties
  prtyObject obj;

  auto intProp = std::make_shared<prtyPropertyUIInfo>(new prtyInt32("testInt", 42), "", "Test Integer");
  auto floatProp = std::make_shared<prtyPropertyUIInfo>(new prtyFloat("testFloat", 3.14f), "", "Test Float");
  auto stringProp = std::make_shared<prtyPropertyUIInfo>(new prtyText("testString", "Hello World"), "", "Test String");
  auto boolProp = std::make_shared<prtyPropertyUIInfo>(new prtyBoolean("testBool", true), "", "Test Boolean");

  obj.AddProperty(intProp);
  obj.AddProperty(floatProp);
  obj.AddProperty(stringProp);
  obj.AddProperty(boolProp);

  // Write to YAML
  docWriterYaml yamlWriter;
  std::string yamlPath = "test_output.yaml";

  yamlWriter.beginDocument(yamlPath);
  writePrtyObject(yamlWriter, &obj, "testObject");
  yamlWriter.endDocument();

  // Verify the file was created
  REQUIRE(std::filesystem::exists(yamlPath));

  // Read and verify contents
  std::ifstream file(yamlPath);
  REQUIRE(file.is_open());

  std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  // Check that the YAML contains expected values
  REQUIRE(content.find("---") != std::string::npos); // YAML header
  REQUIRE(content.find("testObject:") != std::string::npos);
  REQUIRE(content.find("testInt:") != std::string::npos);
  REQUIRE(content.find("42") != std::string::npos);
  REQUIRE(content.find("testFloat:") != std::string::npos);
  REQUIRE(content.find("3.14") != std::string::npos);
  REQUIRE(content.find("testString:") != std::string::npos);
  REQUIRE(content.find("Hello World") != std::string::npos);
  REQUIRE(content.find("testBool:") != std::string::npos);

  // Cleanup
  std::filesystem::remove(yamlPath);
}

TEST_CASE("Serialize AppearanceObject", "[serialize][docWriter][AppearanceObject]")
{
  AppearanceObject appearance;

  // Write to JSON
  docWriterJson jsonWriter;
  std::string jsonPath = "test_appearance.json";

  jsonWriter.beginDocument(jsonPath);
  writePrtyObject(jsonWriter, &appearance, "appearance");
  jsonWriter.endDocument();

  // Verify the file was created
  REQUIRE(std::filesystem::exists(jsonPath));

  // Read and verify contents
  std::ifstream file(jsonPath);
  REQUIRE(file.is_open());

  std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  // Check that the JSON contains the appearance object
  REQUIRE(content.find("appearance") != std::string::npos);

  // Cleanup
  std::filesystem::remove(jsonPath);

  // Write to YAML
  docWriterYaml yamlWriter;
  std::string yamlPath = "test_appearance.yaml";

  yamlWriter.beginDocument(yamlPath);
  writePrtyObject(yamlWriter, &appearance, "appearance");
  yamlWriter.endDocument();

  // Verify the file was created
  REQUIRE(std::filesystem::exists(yamlPath));

  // Cleanup
  std::filesystem::remove(yamlPath);
}

TEST_CASE("Validate nesting protection in docWriter", "[serialize][docWriter]")
{
  SECTION("Extra endObject call")
  {
    docWriterJson writer;
    writer.beginDocument("test_nesting_error.json");
    writer.beginObject("obj1");
    writer.endObject();
    writer.endObject(); // Extra end - should log error but not crash
    writer.endDocument();
    std::filesystem::remove("test_nesting_error.json");
  }

  SECTION("Mismatched begin/end types")
  {
    docWriterJson writer;
    writer.beginDocument("test_mismatch_error.json");
    writer.beginObject("obj1");
    writer.endList(); // Wrong type - should log error
    writer.endDocument();
    std::filesystem::remove("test_mismatch_error.json");
  }

  SECTION("Unclosed contexts")
  {
    docWriterJson writer;
    writer.beginDocument("test_unclosed_error.json");
    writer.beginObject("obj1");
    writer.beginObject("obj2");
    // Missing endObject calls - should log error on endDocument
    writer.endDocument();
    std::filesystem::remove("test_unclosed_error.json");
  }
}
