#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "renderlib/AppearanceObject.hpp"
#include "renderlib/serialize/docReader.h"
#include "renderlib/serialize/docReaderJson.h"
#include "renderlib/serialize/docReaderYaml.h"
#include "renderlib/serialize/docWriter.h"
#include "renderlib/serialize/docWriterJson.h"
#include "renderlib/serialize/docWriterYaml.h"
#include "renderlib/serialize/SerializationConstants.h"
#include "renderlib/Logging.h"

#include <filesystem>
#include <fstream>
#include <string>

// Helper function to create an AppearanceObject with known test values
AppearanceObject*
createTestAppearanceObject()
{
  AppearanceObject* appearance = new AppearanceObject();

  // Set specific test values for all properties
  appearance->appearanceDataObject().RendererType.SetValue(1);
  appearance->appearanceDataObject().ShadingType.SetValue(1);
  appearance->appearanceDataObject().DensityScale.SetValue(2.5f);
  appearance->appearanceDataObject().GradientFactor.SetValue(0.75f);
  appearance->appearanceDataObject().StepSizePrimaryRay.SetValue(1.5f);
  appearance->appearanceDataObject().StepSizeSecondaryRay.SetValue(2.0f);
  appearance->appearanceDataObject().Interpolate.SetValue(true);
  appearance->appearanceDataObject().BackgroundColor.SetValue(glm::vec4(0.2f, 0.3f, 0.4f, 1.0f));
  appearance->appearanceDataObject().ShowBoundingBox.SetValue(true);
  appearance->appearanceDataObject().BoundingBoxColor.SetValue(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
  appearance->appearanceDataObject().ShowScaleBar.SetValue(true);

  return appearance;
}

// Helper function to verify AppearanceObject values match expected test values
void
verifyTestAppearanceObject(AppearanceObject* appearance)
{
  REQUIRE(appearance != nullptr);

  REQUIRE(appearance->appearanceDataObject().RendererType.GetValue() == 1);
  REQUIRE(appearance->appearanceDataObject().ShadingType.GetValue() == 1);
  REQUIRE(appearance->appearanceDataObject().DensityScale.GetValue() == Catch::Approx(2.5f));
  REQUIRE(appearance->appearanceDataObject().GradientFactor.GetValue() == Catch::Approx(0.75f));
  REQUIRE(appearance->appearanceDataObject().StepSizePrimaryRay.GetValue() == Catch::Approx(1.5f));
  REQUIRE(appearance->appearanceDataObject().StepSizeSecondaryRay.GetValue() == Catch::Approx(2.0f));
  REQUIRE(appearance->appearanceDataObject().Interpolate.GetValue() == true);

  auto bgColor = appearance->appearanceDataObject().BackgroundColor.GetValue();
  REQUIRE(bgColor.x == Catch::Approx(0.2f));
  REQUIRE(bgColor.y == Catch::Approx(0.3f));
  REQUIRE(bgColor.z == Catch::Approx(0.4f));
  REQUIRE(bgColor.w == Catch::Approx(1.0f));

  REQUIRE(appearance->appearanceDataObject().ShowBoundingBox.GetValue() == true);

  auto bbColor = appearance->appearanceDataObject().BoundingBoxColor.GetValue();
  REQUIRE(bbColor.x == Catch::Approx(1.0f));
  REQUIRE(bbColor.y == Catch::Approx(0.0f));
  REQUIRE(bbColor.z == Catch::Approx(0.0f));
  REQUIRE(bbColor.w == Catch::Approx(1.0f));

  REQUIRE(appearance->appearanceDataObject().ShowScaleBar.GetValue() == true);
}

TEST_CASE("AppearanceObject JSON roundtrip serialization", "[AppearanceObject][serialize]")
{
  std::string jsonPath = "test_appearance.json";

  // Create and save appearance
  {
    AppearanceObject* appearance = createTestAppearanceObject();

    docWriterJson writer;
    writer.beginDocument(jsonPath);
    appearance->toDocument(&writer);
    writer.endDocument();

    delete appearance;
  }

  // Verify file exists
  REQUIRE(std::filesystem::exists(jsonPath));

  // Load and verify appearance
  {
    AppearanceObject* loadedAppearance = new AppearanceObject();

    docReaderJson reader;
    reader.beginDocument(jsonPath);
    reader.beginObject("appearance0");
    loadedAppearance->fromDocument(&reader);
    reader.endObject();
    reader.endDocument();

    verifyTestAppearanceObject(loadedAppearance);

    delete loadedAppearance;
  }

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("AppearanceObject YAML roundtrip serialization", "[AppearanceObject][serialize]")
{
  std::string yamlPath = "test_appearance.yaml";

  // Create and save appearance
  {
    AppearanceObject* appearance = createTestAppearanceObject();

    docWriterYaml writer;
    writer.beginDocument(yamlPath);
    appearance->toDocument(&writer);
    writer.endDocument();

    delete appearance;
  }

  // Verify file exists
  REQUIRE(std::filesystem::exists(yamlPath));

  // Load and verify appearance
  {
    AppearanceObject* loadedAppearance = new AppearanceObject();

    docReaderYaml reader;
    reader.beginDocument(yamlPath);
    reader.beginObject("appearance0");
    loadedAppearance->fromDocument(&reader);
    reader.endObject();
    reader.endDocument();

    verifyTestAppearanceObject(loadedAppearance);

    delete loadedAppearance;
  }

  // Cleanup
  std::filesystem::remove(yamlPath);
}

TEST_CASE("AppearanceObject version 1 JSON format compatibility", "[AppearanceObject][serialize][version]")
{
  std::string jsonPath = "test_appearance_v1.json";

  // Manually create a version 1 JSON file to simulate old format
  {
    std::ofstream file(jsonPath);
    file << R"({
  "appearance0": {
    "_type": "AppearanceObject",
    "_version": 1,
    "_name": "appearance0",
    "RendererType": 1,
    "ShadingType": 1,
    "DensityScale": 2.5,
    "GradientFactor": 0.75,
    "StepSizePrimaryRay": 1.5,
    "StepSizeSecondaryRay": 2.0,
    "Interpolate": true,
    "BackgroundColor": [0.2, 0.3, 0.4, 1.0],
    "ShowBoundingBox": true,
    "BoundingBoxColor": [1.0, 0.0, 0.0, 1.0],
    "ShowScaleBar": true
  }
})";
    file.close();
  }

  // Load and verify the version 1 format is correctly read
  {
    AppearanceObject* loadedAppearance = new AppearanceObject();

    docReaderJson reader;
    reader.beginDocument(jsonPath);

    reader.beginObject("appearance0");

    // Check version before reading properties
    std::string objectType = reader.peekObjectType();
    REQUIRE(objectType == "AppearanceObject");

    uint32_t version = reader.peekVersion();
    REQUIRE(version == 1);

    loadedAppearance->fromDocument(&reader);

    reader.endObject();

    reader.endDocument();

    verifyTestAppearanceObject(loadedAppearance);

    delete loadedAppearance;
  }

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("AppearanceObject version 1 YAML format compatibility", "[AppearanceObject][serialize][version]")
{
  std::string yamlPath = "test_appearance_v1.yaml";

  // Manually create a version 1 YAML file to simulate old format
  {
    std::ofstream file(yamlPath);
    file << R"(appearance0:
  _type: AppearanceObject
  _version: 1
  _name: appearance0
  RendererType: 1
  ShadingType: 1
  DensityScale: 2.5
  GradientFactor: 0.75
  StepSizePrimaryRay: 1.5
  StepSizeSecondaryRay: 2.0
  Interpolate: true
  BackgroundColor: [0.2, 0.3, 0.4, 1.0]
  ShowBoundingBox: true
  BoundingBoxColor: [1.0, 0.0, 0.0, 1.0]
  ShowScaleBar: true
)";
    file.close();
  }

  // Load and verify the version 1 format is correctly read
  {
    AppearanceObject* loadedAppearance = new AppearanceObject();

    docReaderYaml reader;
    reader.beginDocument(yamlPath);

    reader.beginObject("appearance0");

    // Check version before reading properties
    std::string objectType = reader.peekObjectType();
    REQUIRE(objectType == "AppearanceObject");

    uint32_t version = reader.peekVersion();
    REQUIRE(version == 1);

    loadedAppearance->fromDocument(&reader);

    reader.endObject();

    reader.endDocument();

    verifyTestAppearanceObject(loadedAppearance);

    delete loadedAppearance;
  }

  // Cleanup
  std::filesystem::remove(yamlPath);
}

TEST_CASE("AppearanceObject default values serialization", "[AppearanceObject][serialize]")
{
  std::string jsonPath = "test_appearance_defaults.json";

  // Create appearance with default values (no modifications)
  {
    AppearanceObject* appearance = new AppearanceObject();

    docWriterJson writer;
    writer.beginDocument(jsonPath);
    appearance->toDocument(&writer);
    writer.endDocument();

    delete appearance;
  }

  // Load and verify defaults are preserved
  {
    AppearanceObject* loadedAppearance = new AppearanceObject();
    AppearanceObject* defaultAppearance = new AppearanceObject();

    docReaderJson reader;
    reader.beginDocument(jsonPath);
    reader.beginObject("appearance0");
    loadedAppearance->fromDocument(&reader);
    reader.endObject();
    reader.endDocument();

    // Verify all values match defaults
    REQUIRE(loadedAppearance->appearanceDataObject().RendererType.GetValue() ==
            defaultAppearance->appearanceDataObject().RendererType.GetValue());
    REQUIRE(loadedAppearance->appearanceDataObject().ShadingType.GetValue() ==
            defaultAppearance->appearanceDataObject().ShadingType.GetValue());
    REQUIRE(loadedAppearance->appearanceDataObject().DensityScale.GetValue() ==
            Catch::Approx(defaultAppearance->appearanceDataObject().DensityScale.GetValue()));
    REQUIRE(loadedAppearance->appearanceDataObject().GradientFactor.GetValue() ==
            Catch::Approx(defaultAppearance->appearanceDataObject().GradientFactor.GetValue()));

    delete loadedAppearance;
    delete defaultAppearance;
  }

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("AppearanceObject partial data loading", "[AppearanceObject][serialize]")
{
  std::string jsonPath = "test_appearance_partial.json";

  // Create a JSON file with only some properties
  {
    std::ofstream file(jsonPath);
    file << R"({
  "appearance0": {
    "_type": "AppearanceObject",
    "_version": 1,
    "_name": "appearance0",
    "DensityScale": 3.0,
    "Interpolate": true
  }
})";
    file.close();
  }

  // Load and verify that specified properties are loaded, others remain default
  {
    AppearanceObject* loadedAppearance = new AppearanceObject();
    AppearanceObject* defaultAppearance = new AppearanceObject();

    docReaderJson reader;
    reader.beginDocument(jsonPath);
    reader.beginObject("appearance0");
    loadedAppearance->fromDocument(&reader);
    reader.endObject();
    reader.endDocument();

    // Specified properties should be loaded
    REQUIRE(loadedAppearance->appearanceDataObject().DensityScale.GetValue() == Catch::Approx(3.0f));
    REQUIRE(loadedAppearance->appearanceDataObject().Interpolate.GetValue() == true);

    // Other properties should remain at default
    REQUIRE(loadedAppearance->appearanceDataObject().RendererType.GetValue() ==
            defaultAppearance->appearanceDataObject().RendererType.GetValue());
    REQUIRE(loadedAppearance->appearanceDataObject().GradientFactor.GetValue() ==
            Catch::Approx(defaultAppearance->appearanceDataObject().GradientFactor.GetValue()));

    delete loadedAppearance;
    delete defaultAppearance;
  }

  // Cleanup
  std::filesystem::remove(jsonPath);
}
