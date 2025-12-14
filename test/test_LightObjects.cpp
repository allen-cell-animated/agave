#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "renderlib/AreaLightObject.hpp"
#include "renderlib/SkyLightObject.hpp"
#include "renderlib/serialize/docReader.h"
#include "renderlib/serialize/docReaderJson.h"
#include "renderlib/serialize/docReaderYaml.h"
#include "renderlib/serialize/docWriter.h"
#include "renderlib/serialize/docWriterJson.h"
#include "renderlib/serialize/docWriterYaml.h"
#include "renderlib/Logging.h"

#include <filesystem>
#include <glm/glm.hpp>

// ============================================================================
// AreaLightObject Tests
// ============================================================================

// Helper function to create an AreaLightObject with known test values
AreaLightObject*
createTestAreaLightObject()
{
  AreaLightObject* light = new AreaLightObject();

  // Set specific test values for all properties
  light->getDataObject().Theta.SetValue(45.0f);
  light->getDataObject().Phi.SetValue(90.0f);
  light->getDataObject().Size.SetValue(2.5f);
  light->getDataObject().Distance.SetValue(15.0f);
  light->getDataObject().Intensity.SetValue(250.0f);
  light->getDataObject().Color.SetValue(glm::vec4(0.8f, 0.9f, 1.0f, 1.0f));

  return light;
}

// Helper function to verify AreaLightObject values match expected test values
void
verifyTestAreaLightObject(AreaLightObject* light)
{
  REQUIRE(light != nullptr);

  const ArealightDataObject& data = light->getDataObject();
  REQUIRE(data.Theta.GetValue() == Catch::Approx(45.0f));
  REQUIRE(data.Phi.GetValue() == Catch::Approx(90.0f));
  REQUIRE(data.Size.GetValue() == Catch::Approx(2.5f));
  REQUIRE(data.Distance.GetValue() == Catch::Approx(15.0f));
  REQUIRE(data.Intensity.GetValue() == Catch::Approx(250.0f));

  glm::vec4 color = data.Color.GetValue();
  REQUIRE(color.x == Catch::Approx(0.8f));
  REQUIRE(color.y == Catch::Approx(0.9f));
  REQUIRE(color.z == Catch::Approx(1.0f));
  REQUIRE(color.w == Catch::Approx(1.0f));
}

TEST_CASE("AreaLightObject creation and initialization", "[AreaLightObject]")
{
  AreaLightObject* light = new AreaLightObject();

  REQUIRE(light != nullptr);
  REQUIRE(light->getSceneLight() != nullptr);

  // Verify default values
  const ArealightDataObject& data = light->getDataObject();
  REQUIRE(data.Theta.GetValue() == Catch::Approx(0.0f));
  REQUIRE(data.Phi.GetValue() == Catch::Approx(0.0f));
  REQUIRE(data.Size.GetValue() == Catch::Approx(1.0f));
  REQUIRE(data.Distance.GetValue() == Catch::Approx(10.0f));
  REQUIRE(data.Intensity.GetValue() == Catch::Approx(100.0f));

  delete light;
}

TEST_CASE("AreaLightObject property setting", "[AreaLightObject]")
{
  AreaLightObject* light = new AreaLightObject();

  // Test setting properties
  light->getDataObject().Theta.SetValue(30.0f);
  light->getDataObject().Phi.SetValue(60.0f);
  light->getDataObject().Size.SetValue(5.0f);
  light->getDataObject().Distance.SetValue(20.0f);
  light->getDataObject().Intensity.SetValue(150.0f);
  light->getDataObject().Color.SetValue(glm::vec4(1.0f, 0.5f, 0.25f, 1.0f));

  REQUIRE(light->getDataObject().Theta.GetValue() == Catch::Approx(30.0f));
  REQUIRE(light->getDataObject().Phi.GetValue() == Catch::Approx(60.0f));
  REQUIRE(light->getDataObject().Size.GetValue() == Catch::Approx(5.0f));
  REQUIRE(light->getDataObject().Distance.GetValue() == Catch::Approx(20.0f));
  REQUIRE(light->getDataObject().Intensity.GetValue() == Catch::Approx(150.0f));

  glm::vec4 color = light->getDataObject().Color.GetValue();
  REQUIRE(color.x == Catch::Approx(1.0f));
  REQUIRE(color.y == Catch::Approx(0.5f));
  REQUIRE(color.z == Catch::Approx(0.25f));

  delete light;
}

TEST_CASE("AreaLightObject JSON roundtrip serialization", "[AreaLightObject][serialize]")
{
  std::string jsonPath = "test_arealight.json";

  // Create and save light
  {
    AreaLightObject* light = createTestAreaLightObject();

    docWriterJson writer;
    writer.beginDocument(jsonPath);
    light->toDocument(&writer);
    writer.endDocument();

    delete light;
  }

  // Verify file exists
  REQUIRE(std::filesystem::exists(jsonPath));

  // Load and verify light
  {
    AreaLightObject* loadedLight = new AreaLightObject();

    docReaderJson reader;
    reader.beginDocument(jsonPath);
    reader.beginObject("areaLight0");
    loadedLight->fromDocument(&reader);
    reader.endObject();
    reader.endDocument();

    verifyTestAreaLightObject(loadedLight);

    delete loadedLight;
  }

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("AreaLightObject YAML roundtrip serialization", "[AreaLightObject][serialize]")
{
  std::string yamlPath = "test_arealight.yaml";

  // Create and save light
  {
    AreaLightObject* light = createTestAreaLightObject();

    docWriterYaml writer;
    writer.beginDocument(yamlPath);
    light->toDocument(&writer);
    writer.endDocument();

    delete light;
  }

  // Verify file exists
  REQUIRE(std::filesystem::exists(yamlPath));

  // Load and verify light
  {
    AreaLightObject* loadedLight = new AreaLightObject();

    docReaderYaml reader;
    reader.beginDocument(yamlPath);
    reader.beginObject("areaLight0");
    loadedLight->fromDocument(&reader);
    reader.endObject();
    reader.endDocument();

    verifyTestAreaLightObject(loadedLight);

    delete loadedLight;
  }

  // Cleanup
  std::filesystem::remove(yamlPath);
}

TEST_CASE("AreaLightObject dirty callback", "[AreaLightObject]")
{
  AreaLightObject* light = new AreaLightObject();
  bool callbackCalled = false;

  light->setDirtyCallback([&callbackCalled]() { callbackCalled = true; });

  // The callback should be invoked when properties change
  light->ThetaChanged(nullptr, false);
  REQUIRE(callbackCalled);

  callbackCalled = false;
  light->PhiChanged(nullptr, false);
  REQUIRE(callbackCalled);

  callbackCalled = false;
  light->IntensityChanged(nullptr, false);
  REQUIRE(callbackCalled);

  delete light;
}

// ============================================================================
// SkyLightObject Tests
// ============================================================================

// Helper function to create a SkyLightObject with known test values
SkyLightObject*
createTestSkyLightObject()
{
  SkyLightObject* light = new SkyLightObject();

  // Set specific test values for all properties
  light->getDataObject().TopIntensity.SetValue(2.5f);
  light->getDataObject().TopColor.SetValue(glm::vec4(0.9f, 0.95f, 1.0f, 1.0f));
  light->getDataObject().MiddleIntensity.SetValue(1.5f);
  light->getDataObject().MiddleColor.SetValue(glm::vec4(0.6f, 0.6f, 0.6f, 1.0f));
  light->getDataObject().BottomIntensity.SetValue(0.8f);
  light->getDataObject().BottomColor.SetValue(glm::vec4(0.3f, 0.25f, 0.2f, 1.0f));

  return light;
}

// Helper function to verify SkyLightObject values match expected test values
void
verifyTestSkyLightObject(SkyLightObject* light)
{
  REQUIRE(light != nullptr);

  const SkylightDataObject& data = light->getDataObject();
  REQUIRE(data.TopIntensity.GetValue() == Catch::Approx(2.5f));
  REQUIRE(data.MiddleIntensity.GetValue() == Catch::Approx(1.5f));
  REQUIRE(data.BottomIntensity.GetValue() == Catch::Approx(0.8f));

  glm::vec4 topColor = data.TopColor.GetValue();
  REQUIRE(topColor.x == Catch::Approx(0.9f));
  REQUIRE(topColor.y == Catch::Approx(0.95f));
  REQUIRE(topColor.z == Catch::Approx(1.0f));

  glm::vec4 middleColor = data.MiddleColor.GetValue();
  REQUIRE(middleColor.x == Catch::Approx(0.6f));
  REQUIRE(middleColor.y == Catch::Approx(0.6f));
  REQUIRE(middleColor.z == Catch::Approx(0.6f));

  glm::vec4 bottomColor = data.BottomColor.GetValue();
  REQUIRE(bottomColor.x == Catch::Approx(0.3f));
  REQUIRE(bottomColor.y == Catch::Approx(0.25f));
  REQUIRE(bottomColor.z == Catch::Approx(0.2f));
}

TEST_CASE("SkyLightObject creation and initialization", "[SkyLightObject]")
{
  SkyLightObject* light = new SkyLightObject();

  REQUIRE(light != nullptr);
  REQUIRE(light->getSceneLight() != nullptr);

  // Verify default values
  const SkylightDataObject& data = light->getDataObject();
  REQUIRE(data.TopIntensity.GetValue() == Catch::Approx(1.0f));
  REQUIRE(data.MiddleIntensity.GetValue() == Catch::Approx(1.0f));
  REQUIRE(data.BottomIntensity.GetValue() == Catch::Approx(1.0f));

  glm::vec4 topColor = data.TopColor.GetValue();
  REQUIRE(topColor.x == Catch::Approx(1.0f));
  REQUIRE(topColor.y == Catch::Approx(1.0f));
  REQUIRE(topColor.z == Catch::Approx(1.0f));

  glm::vec4 bottomColor = data.BottomColor.GetValue();
  REQUIRE(bottomColor.x == Catch::Approx(0.2f));
  REQUIRE(bottomColor.y == Catch::Approx(0.2f));
  REQUIRE(bottomColor.z == Catch::Approx(0.2f));

  delete light;
}

TEST_CASE("SkyLightObject property setting", "[SkyLightObject]")
{
  SkyLightObject* light = new SkyLightObject();

  // Test setting properties
  light->getDataObject().TopIntensity.SetValue(3.0f);
  light->getDataObject().TopColor.SetValue(glm::vec4(1.0f, 1.0f, 0.8f, 1.0f));
  light->getDataObject().MiddleIntensity.SetValue(2.0f);
  light->getDataObject().MiddleColor.SetValue(glm::vec4(0.7f, 0.7f, 0.7f, 1.0f));
  light->getDataObject().BottomIntensity.SetValue(1.0f);
  light->getDataObject().BottomColor.SetValue(glm::vec4(0.4f, 0.3f, 0.2f, 1.0f));

  REQUIRE(light->getDataObject().TopIntensity.GetValue() == Catch::Approx(3.0f));
  REQUIRE(light->getDataObject().MiddleIntensity.GetValue() == Catch::Approx(2.0f));
  REQUIRE(light->getDataObject().BottomIntensity.GetValue() == Catch::Approx(1.0f));

  glm::vec4 topColor = light->getDataObject().TopColor.GetValue();
  REQUIRE(topColor.x == Catch::Approx(1.0f));
  REQUIRE(topColor.y == Catch::Approx(1.0f));
  REQUIRE(topColor.z == Catch::Approx(0.8f));

  delete light;
}

TEST_CASE("SkyLightObject JSON roundtrip serialization", "[SkyLightObject][serialize]")
{
  std::string jsonPath = "test_skylight.json";

  // Create and save light
  {
    SkyLightObject* light = createTestSkyLightObject();

    docWriterJson writer;
    writer.beginDocument(jsonPath);
    light->toDocument(&writer);
    writer.endDocument();

    delete light;
  }

  // Verify file exists
  REQUIRE(std::filesystem::exists(jsonPath));

  // Load and verify light
  {
    SkyLightObject* loadedLight = new SkyLightObject();

    docReaderJson reader;
    reader.beginDocument(jsonPath);
    reader.beginObject("skyLight0");
    loadedLight->fromDocument(&reader);
    reader.endObject();
    reader.endDocument();

    verifyTestSkyLightObject(loadedLight);

    delete loadedLight;
  }

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("SkyLightObject YAML roundtrip serialization", "[SkyLightObject][serialize]")
{
  std::string yamlPath = "test_skylight.yaml";

  // Create and save light
  {
    SkyLightObject* light = createTestSkyLightObject();

    docWriterYaml writer;
    writer.beginDocument(yamlPath);
    light->toDocument(&writer);
    writer.endDocument();

    delete light;
  }

  // Verify file exists
  REQUIRE(std::filesystem::exists(yamlPath));

  // Load and verify light
  {
    SkyLightObject* loadedLight = new SkyLightObject();

    docReaderYaml reader;
    reader.beginDocument(yamlPath);
    reader.beginObject("skyLight0");
    loadedLight->fromDocument(&reader);
    reader.endObject();
    reader.endDocument();

    verifyTestSkyLightObject(loadedLight);

    delete loadedLight;
  }

  // Cleanup
  std::filesystem::remove(yamlPath);
}

TEST_CASE("SkyLightObject dirty callback", "[SkyLightObject]")
{
  SkyLightObject* light = new SkyLightObject();
  bool callbackCalled = false;

  light->setDirtyCallback([&callbackCalled]() { callbackCalled = true; });

  // The callback should be invoked when properties change
  light->TopIntensityChanged(nullptr, false);
  REQUIRE(callbackCalled);

  callbackCalled = false;
  light->TopColorChanged(nullptr, false);
  REQUIRE(callbackCalled);

  callbackCalled = false;
  light->MiddleIntensityChanged(nullptr, false);
  REQUIRE(callbackCalled);

  callbackCalled = false;
  light->BottomColorChanged(nullptr, false);
  REQUIRE(callbackCalled);

  delete light;
}

TEST_CASE("SkyLightObject and AreaLightObject independent instances", "[LightObjects]")
{
  AreaLightObject* areaLight = new AreaLightObject();
  SkyLightObject* skyLight = new SkyLightObject();

  // Verify they have different scene lights
  REQUIRE(areaLight->getSceneLight() != skyLight->getSceneLight());

  // Modify one and verify the other is unaffected
  areaLight->getDataObject().Intensity.SetValue(500.0f);
  REQUIRE(skyLight->getDataObject().TopIntensity.GetValue() == Catch::Approx(1.0f));

  skyLight->getDataObject().TopIntensity.SetValue(5.0f);
  REQUIRE(areaLight->getDataObject().Intensity.GetValue() == Catch::Approx(500.0f));

  delete areaLight;
  delete skyLight;
}
