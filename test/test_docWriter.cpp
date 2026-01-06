#include <catch2/catch_test_macros.hpp>

#include "renderlib/serialize/docWriter.h"
#include "renderlib/serialize/docWriterJson.h"
#include "renderlib/serialize/docWriterYaml.h"
#include "renderlib/AppearanceObject.hpp"
#include "renderlib/core/prty/prtyObject.hpp"
#include "renderlib/core/prty/prtyPropertyUIInfo.hpp"
#include "renderlib/core/prty/prtyProperty.hpp"
#include "renderlib/core/prty/prtyIntegerTemplate.hpp"
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

  writer.beginObject(name.c_str(), "MYTYPE", 1);

  // Write properties directly
  for (const auto& propUIInfo : obj->GetList()) {
    int numProps = propUIInfo->GetNumberOfProperties();
    for (int i = 0; i < numProps; ++i) {
      prtyProperty* prop = propUIInfo->GetProperty(i);
      if (prop) {
        prop->Write(writer);
      }
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
    writer.beginObject("obj1", "MYTYPE", 1);
    writer.endObject();
    writer.endObject(); // Extra end - should log error but not crash
    writer.endDocument();
    std::filesystem::remove("test_nesting_error.json");
  }

  SECTION("Mismatched begin/end types")
  {
    docWriterJson writer;
    writer.beginDocument("test_mismatch_error.json");
    writer.beginObject("obj1", "MYTYPE", 1);
    writer.endList(); // Wrong type - should log error
    writer.endDocument();
    std::filesystem::remove("test_mismatch_error.json");
  }

  SECTION("Unclosed contexts")
  {
    docWriterJson writer;
    writer.beginDocument("test_unclosed_error.json");
    writer.beginObject("obj1", "MYTYPE", 1);
    writer.beginObject("obj2", "MYTYPE", 1);
    // Missing endObject calls - should log error on endDocument
    writer.endDocument();
    std::filesystem::remove("test_unclosed_error.json");
  }
}
