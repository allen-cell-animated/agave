#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "renderlib/CameraObject.hpp"
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

// Helper function to create a CameraObject with known test values
CameraObject*
createTestCameraObject()
{
  CameraObject* camera = new CameraObject();

  // Set specific test values for all properties
  camera->getCameraDataObject().Exposure.SetValue(0.85f);
  camera->getCameraDataObject().ExposureIterations.SetValue(2); // "4" in enum
  camera->getCameraDataObject().NoiseReduction.SetValue(true);
  camera->getCameraDataObject().ApertureSize.SetValue(0.05f);
  camera->getCameraDataObject().FieldOfView.SetValue(45.0f);
  camera->getCameraDataObject().FocalDistance.SetValue(5.5f);
  camera->getCameraDataObject().Position.SetValue(glm::vec3(1.0f, 2.0f, 3.0f));
  camera->getCameraDataObject().Target.SetValue(glm::vec3(0.0f, 0.0f, 0.0f));
  camera->getCameraDataObject().NearPlane.SetValue(0.5f);
  camera->getCameraDataObject().FarPlane.SetValue(500.0f);
  camera->getCameraDataObject().Roll.SetValue(15.0f);
  camera->getCameraDataObject().OrthoScale.SetValue(1.5f);
  camera->getCameraDataObject().ProjectionMode.SetValue(0); // Perspective

  return camera;
}

// Helper function to verify CameraObject values match expected test values
void
verifyTestCameraObject(CameraObject* camera)
{
  REQUIRE(camera != nullptr);

  REQUIRE(camera->getCameraDataObject().Exposure.GetValue() == Catch::Approx(0.85f));
  REQUIRE(camera->getCameraDataObject().ExposureIterations.GetValue() == 2);
  REQUIRE(camera->getCameraDataObject().NoiseReduction.GetValue() == true);
  REQUIRE(camera->getCameraDataObject().ApertureSize.GetValue() == Catch::Approx(0.05f));
  REQUIRE(camera->getCameraDataObject().FieldOfView.GetValue() == Catch::Approx(45.0f));
  REQUIRE(camera->getCameraDataObject().FocalDistance.GetValue() == Catch::Approx(5.5f));

  auto pos = camera->getCameraDataObject().Position.GetValue();
  REQUIRE(pos.x == Catch::Approx(1.0f));
  REQUIRE(pos.y == Catch::Approx(2.0f));
  REQUIRE(pos.z == Catch::Approx(3.0f));

  auto target = camera->getCameraDataObject().Target.GetValue();
  REQUIRE(target.x == Catch::Approx(0.0f));
  REQUIRE(target.y == Catch::Approx(0.0f));
  REQUIRE(target.z == Catch::Approx(0.0f));

  REQUIRE(camera->getCameraDataObject().NearPlane.GetValue() == Catch::Approx(0.5f));
  REQUIRE(camera->getCameraDataObject().FarPlane.GetValue() == Catch::Approx(500.0f));
  REQUIRE(camera->getCameraDataObject().Roll.GetValue() == Catch::Approx(15.0f));
  REQUIRE(camera->getCameraDataObject().OrthoScale.GetValue() == Catch::Approx(1.5f));
  REQUIRE(camera->getCameraDataObject().ProjectionMode.GetValue() == 0);
}

TEST_CASE("CameraObject JSON roundtrip serialization", "[CameraObject][serialize]")
{
  std::string jsonPath = "test_camera.json";

  // Create and save camera
  {
    CameraObject* camera = createTestCameraObject();

    docWriterJson writer;
    writer.beginDocument(jsonPath);
    camera->toDocument(&writer);
    writer.endDocument();

    delete camera;
  }

  // Verify file exists
  REQUIRE(std::filesystem::exists(jsonPath));

  // Load and verify camera
  {
    CameraObject* loadedCamera = new CameraObject();

    docReaderJson reader;
    reader.beginDocument(jsonPath);
    reader.beginObject("camera0");
    loadedCamera->fromDocument(&reader);
    reader.endObject();
    reader.endDocument();

    verifyTestCameraObject(loadedCamera);

    delete loadedCamera;
  }

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("CameraObject YAML roundtrip serialization", "[CameraObject][serialize]")
{
  std::string yamlPath = "test_camera.yaml";

  // Create and save camera
  {
    CameraObject* camera = createTestCameraObject();

    docWriterYaml writer;
    writer.beginDocument(yamlPath);
    camera->toDocument(&writer);
    writer.endDocument();

    delete camera;
  }

  // Verify file exists
  REQUIRE(std::filesystem::exists(yamlPath));

  // Load and verify camera
  {
    CameraObject* loadedCamera = new CameraObject();

    docReaderYaml reader;
    reader.beginDocument(yamlPath);
    reader.beginObject("camera0");
    loadedCamera->fromDocument(&reader);
    reader.endObject();
    reader.endDocument();

    verifyTestCameraObject(loadedCamera);

    delete loadedCamera;
  }

  // Cleanup
  std::filesystem::remove(yamlPath);
}

TEST_CASE("CameraObject version 1 JSON format compatibility", "[CameraObject][serialize][version]")
{
  std::string jsonPath = "test_camera_v1.json";

  // Manually create a version 1 JSON file to simulate old format
  {
    std::ofstream file(jsonPath);
    file << R"({
  "camera0": {
    "_type": "CameraObject",
    "_version": 1,
    "_name": "camera0",
    "Exposure": 0.85,
    "ExposureIterations": 2,
    "NoiseReduction": true,
    "ApertureSize": 0.05,
    "FieldOfView": 45.0,
    "FocalDistance": 5.5,
    "Position": [1.0, 2.0, 3.0],
    "Target": [0.0, 0.0, 0.0],
    "NearPlane": 0.5,
    "FarPlane": 500.0,
    "Roll": 15.0,
    "OrthoScale": 1.5,
    "ProjectionMode": 0
  }
})";
    file.close();
  }

  // Load and verify the version 1 format is correctly read
  {
    CameraObject* loadedCamera = new CameraObject();

    docReaderJson reader;
    reader.beginDocument(jsonPath);

    reader.beginObject("camera0");

    // Check version before reading properties
    std::string objectType = reader.peekObjectType();
    REQUIRE(objectType == "CameraObject");

    int version = reader.peekVersion();
    REQUIRE(version == 1);

    loadedCamera->fromDocument(&reader);

    reader.endObject();

    reader.endDocument();

    verifyTestCameraObject(loadedCamera);

    delete loadedCamera;
  }

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("CameraObject version 1 YAML format compatibility", "[CameraObject][serialize][version]")
{
  std::string yamlPath = "test_camera_v1.yaml";

  // Manually create a version 1 YAML file to simulate old format
  {
    std::ofstream file(yamlPath);
    file << R"(camera0:
  _type: CameraObject
  _version: 1
  _name: camera0
  Exposure: 0.85
  ExposureIterations: 2
  NoiseReduction: true
  ApertureSize: 0.05
  FieldOfView: 45.0
  FocalDistance: 5.5
  Position: [1.0, 2.0, 3.0]
  Target: [0.0, 0.0, 0.0]
  NearPlane: 0.5
  FarPlane: 500.0
  Roll: 15.0
  OrthoScale: 1.5
  ProjectionMode: 0
)";
    file.close();
  }

  // Load and verify the version 1 format is correctly read
  {
    CameraObject* loadedCamera = new CameraObject();

    docReaderYaml reader;
    reader.beginDocument(yamlPath);

    reader.beginObject("camera0");

    // Check version before reading properties
    std::string objectType = reader.peekObjectType();
    REQUIRE(objectType == "CameraObject");

    int version = reader.peekVersion();
    REQUIRE(version == 1);

    loadedCamera->fromDocument(&reader);

    reader.endObject();

    reader.endDocument();

    verifyTestCameraObject(loadedCamera);

    delete loadedCamera;
  }

  // Cleanup
  std::filesystem::remove(yamlPath);
}

TEST_CASE("CameraObject default values serialization", "[CameraObject][serialize]")
{
  std::string jsonPath = "test_camera_defaults.json";

  // Create camera with default values (no modifications)
  {
    CameraObject* camera = new CameraObject();

    docWriterJson writer;
    writer.beginDocument(jsonPath);
    camera->toDocument(&writer);
    writer.endDocument();

    delete camera;
  }

  // Load and verify defaults are preserved
  {
    CameraObject* loadedCamera = new CameraObject();
    CameraObject* defaultCamera = new CameraObject();

    docReaderJson reader;
    reader.beginDocument(jsonPath);
    reader.beginObject("camera0");
    loadedCamera->fromDocument(&reader);
    reader.endObject();
    reader.endDocument();

    // Verify all values match defaults
    REQUIRE(loadedCamera->getCameraDataObject().Exposure.GetValue() ==
            Catch::Approx(defaultCamera->getCameraDataObject().Exposure.GetValue()));
    REQUIRE(loadedCamera->getCameraDataObject().ExposureIterations.GetValue() ==
            defaultCamera->getCameraDataObject().ExposureIterations.GetValue());
    REQUIRE(loadedCamera->getCameraDataObject().NoiseReduction.GetValue() ==
            defaultCamera->getCameraDataObject().NoiseReduction.GetValue());
    REQUIRE(loadedCamera->getCameraDataObject().ApertureSize.GetValue() ==
            Catch::Approx(defaultCamera->getCameraDataObject().ApertureSize.GetValue()));
    REQUIRE(loadedCamera->getCameraDataObject().FieldOfView.GetValue() ==
            Catch::Approx(defaultCamera->getCameraDataObject().FieldOfView.GetValue()));

    delete loadedCamera;
    delete defaultCamera;
  }

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("CameraObject partial data loading", "[CameraObject][serialize]")
{
  std::string jsonPath = "test_camera_partial.json";

  // Create a JSON file with only some properties
  {
    std::ofstream file(jsonPath);
    file << R"({
  "camera0": {
    "_type": "CameraObject",
    "_version": 1,
    "_name": "camera0",
    "Exposure": 0.5,
    "FieldOfView": 60.0
  }
})";
    file.close();
  }

  // Load and verify that specified properties are loaded, others remain default
  {
    CameraObject* loadedCamera = new CameraObject();
    CameraObject* defaultCamera = new CameraObject();

    docReaderJson reader;
    reader.beginDocument(jsonPath);
    reader.beginObject("camera0");
    loadedCamera->fromDocument(&reader);
    reader.endObject();
    reader.endDocument();

    // Specified properties should be loaded
    REQUIRE(loadedCamera->getCameraDataObject().Exposure.GetValue() == Catch::Approx(0.5f));
    REQUIRE(loadedCamera->getCameraDataObject().FieldOfView.GetValue() == Catch::Approx(60.0f));

    // Other properties should remain at default
    REQUIRE(loadedCamera->getCameraDataObject().ExposureIterations.GetValue() ==
            defaultCamera->getCameraDataObject().ExposureIterations.GetValue());
    REQUIRE(loadedCamera->getCameraDataObject().ApertureSize.GetValue() ==
            Catch::Approx(defaultCamera->getCameraDataObject().ApertureSize.GetValue()));

    delete loadedCamera;
    delete defaultCamera;
  }

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("CameraObject invalid version handling", "[CameraObject][serialize][version]")
{
  std::string jsonPath = "test_camera_invalid_version.json";

  // Create a JSON file with a future version number
  {
    std::ofstream file(jsonPath);
    file << R"({
  "camera0": {
    "_type": "CameraObject",
    "_version": 999,
    "_name": "camera0",
    "Exposure": 0.85,
    "FieldOfView": 45.0
  }
})";
    file.close();
  }

  // Load - should still work, just log a warning
  {
    CameraObject* loadedCamera = new CameraObject();

    docReaderJson reader;
    reader.beginDocument(jsonPath);

    // Enter the CameraObject before checking version
    reader.beginObject("camera0");

    uint32_t version = reader.peekVersion();
    REQUIRE(version == 999);

    loadedCamera->fromDocument(&reader);

    reader.endObject();

    reader.endDocument();

    REQUIRE(loadedCamera->getCameraDataObject().Exposure.GetValue() == Catch::Approx(0.85f));
    REQUIRE(loadedCamera->getCameraDataObject().FieldOfView.GetValue() == Catch::Approx(45.0f));

    delete loadedCamera;
  }

  // Cleanup
  std::filesystem::remove(jsonPath);
}

TEST_CASE("CameraObject enum property serialization", "[CameraObject][serialize]")
{
  std::string jsonPath = "test_camera_enum.json";

  // Test different enum values
  for (int i = 0; i < 4; ++i) {
    // Create and save camera with specific enum value
    {
      CameraObject* camera = new CameraObject();
      camera->getCameraDataObject().ExposureIterations.SetValue(i);

      docWriterJson writer;
      writer.beginDocument(jsonPath);
      camera->toDocument(&writer);
      writer.endDocument();

      delete camera;
    }

    // Load and verify enum value
    {
      CameraObject* loadedCamera = new CameraObject();

      docReaderJson reader;
      reader.beginDocument(jsonPath);
      reader.beginObject("camera0");
      loadedCamera->fromDocument(&reader);
      reader.endObject();
      reader.endDocument();

      REQUIRE(loadedCamera->getCameraDataObject().ExposureIterations.GetValue() == i);

      delete loadedCamera;
    }
  }

  // Test ProjectionMode enum
  for (int i = 0; i < 2; ++i) {
    // Create and save camera with specific projection mode
    {
      CameraObject* camera = new CameraObject();
      camera->getCameraDataObject().ProjectionMode.SetValue(i);

      docWriterJson writer;
      writer.beginDocument(jsonPath);
      camera->toDocument(&writer);
      writer.endDocument();

      delete camera;
    }

    // Load and verify projection mode
    {
      CameraObject* loadedCamera = new CameraObject();

      docReaderJson reader;
      reader.beginDocument(jsonPath);
      reader.beginObject("camera0");
      loadedCamera->fromDocument(&reader);
      reader.endObject();
      reader.endDocument();

      REQUIRE(loadedCamera->getCameraDataObject().ProjectionMode.GetValue() == i);

      delete loadedCamera;
    }
  }

  // Cleanup
  std::filesystem::remove(jsonPath);
}
