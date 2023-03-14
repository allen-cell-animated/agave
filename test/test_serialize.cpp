#include <catch2/catch_test_macros.hpp>

#include "../agave_app/Serialize.h"
#include "renderlib/json/json.hpp"

// #include <array>
// #include <vector>

TEST_CASE("Json Serialization", "[serialize]")
{
  SECTION("Read and write PathTraceSettings_V1")
  {
    auto settings = PathTraceSettings_V1{};
    settings.primaryStepSize = 0.1f;
    settings.secondaryStepSize = 0.2f;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<PathTraceSettings_V1>();
    REQUIRE(settings == settings2);
  }
  SECTION("Read and write TimelineSettings_V1")
  {
    auto settings = TimelineSettings_V1{};
    settings.minTime = 0.1f;
    settings.maxTime = 0.2f;
    settings.currentTime = 0.3f;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<TimelineSettings_V1>();
    REQUIRE(settings == settings2);
  }
  SECTION("Read and write CameraSettings_V1")
  {
    auto settings = CameraSettings_V1{};
    settings.eye = { 0.1f, 0.2f, 0.3f };
    settings.target = { 0.4f, 0.5f, 0.6f };
    settings.up = { 0.7f, 0.8f, 0.9f };
    settings.projection = Projection::PERSPECTIVE;
    settings.fovY = 0.1f;
    settings.orthoScale = 0.2f;
    settings.exposure = 0.3f;
    settings.aperture = 0.4f;
    settings.focalDistance = 0.5f;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<CameraSettings_V1>();
    REQUIRE(settings == settings2);
  }

  SECTION("Read and write ControlPointSettings_V1")
  {
    auto settings = ControlPointSettings_V1{};
    settings.x = 0.1f;
    settings.value = { 0.2f, 0.3f, 0.4f, 0.5f };
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<ControlPointSettings_V1>();
    REQUIRE(settings == settings2);
  }

  SECTION("Read and write LutParams_V1")
  {
    auto settings = LutParams_V1{};
    settings.window = 0.1f;
    settings.level = 0.2f;
    settings.isovalue = 0.3f;
    settings.isorange = 0.4f;
    settings.pctLow = 0.5f;
    settings.pctHigh = 0.6f;
    settings.controlPoints = { { 0.1f, { 0.2f, 0.3f, 0.4f, 0.5f } }, { 0.2f, { 0.3f, 0.4f, 0.5f, 0.6f } } };
    settings.mode = GradientEditMode::CUSTOM;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<LutParams_V1>();
    REQUIRE(settings == settings2);
  }

  SECTION("Read and write ChannelSettings_V1")
  {
    auto settings = ChannelSettings_V1{};
    settings.enabled = true;
    settings.lutParams = { 0.1f,
                           0.2f,
                           0.3f,
                           0.4f,
                           0.5f,
                           0.6f,
                           { { 0.1f, { 0.2f, 0.3f, 0.4f, 0.5f } }, { 0.2f, { 0.3f, 0.4f, 0.5f, 0.6f } } },
                           GradientEditMode::CUSTOM };
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<ChannelSettings_V1>();
    REQUIRE(settings == settings2);
  }

  SECTION("Read and write LightSettings_V1")
  {
    auto settings = LightSettings_V1{};
    settings.type = LightType::AREA;
    settings.distance = 0.1;
    settings.theta = 0.2;
    settings.phi = 0.3;
    settings.color = { 0.1f, 0.2f, 0.3f };
    settings.colorIntensity = 0.4f;
    settings.topColor = { 0.5f, 0.6f, 0.7f };
    settings.topColorIntensity = 0.8f;
    settings.middleColor = { 0.9f, 0.1f, 0.2f };
    settings.middleColorIntensity = 0.3f;
    settings.bottomColor = { 0.9f, 0.1f, 0.2f };
    settings.bottomColorIntensity = 0.3f;
    settings.width = 0.4f;
    settings.height = 0.5f;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<LightSettings_V1>();
    REQUIRE(settings == settings2);
  }

  SECTION("Read and write CaptureSettings_V1")
  {
    auto settings = CaptureSettings_V1{};
    settings.width = 1;
    settings.height = 2;
    settings.filenamePrefix = "test";
    settings.outputDirectory = "test2";
    settings.samples = 3;
    settings.seconds = 4.0;
    settings.durationType = DurationType::TIME;
    settings.startTime = 6;
    settings.endTime = 7;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<CaptureSettings_V1>();
    REQUIRE(settings == settings2);
  }

  SECTION("Read and write ViewerState_V1")
  {
    auto settings = ViewerState_V1{};
    settings.name = "test";
    settings.version = { 1, 2, 3 };
    settings.resolution = { 1, 2 };
    settings.renderIterations = 3;
    settings.pathTracer = PathTraceSettings_V1{
      0.1f,
      0.2f,
    };
    settings.timeline = TimelineSettings_V1{ 0.1, 0.2, 0.3 };
    settings.scene = 1;
    settings.clipRegion = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f };
    settings.scale = { 0.1f, 0.2f, 0.3f };
    settings.camera = CameraSettings_V1{};
    settings.backgroundColor = { 0.1f, 0.2f, 0.3f };
    settings.showBoundingBox = true;
    settings.channels = { ChannelSettings_V1{}, ChannelSettings_V1{} };
    settings.density = 100.0;
    settings.lights = { LightSettings_V1{}, LightSettings_V1{} };
    settings.capture = CaptureSettings_V1{};
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<ViewerState_V1>();
    REQUIRE(settings == settings2);
  }
}
