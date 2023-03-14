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
    REQUIRE(settings.primaryStepSize == settings2.primaryStepSize);
    REQUIRE(settings.secondaryStepSize == settings2.secondaryStepSize);
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
    REQUIRE(settings.minTime == settings2.minTime);
    REQUIRE(settings.maxTime == settings2.maxTime);
    REQUIRE(settings.currentTime == settings2.currentTime);
  }
  SECTION("Read and write CameraSettings_V1")
  {
    auto settings = CameraSettings_V1{};
    settings.eye = { 0.1f, 0.2f, 0.3f };
    settings.target = { 0.4f, 0.5f, 0.6f };
    settings.up = { 0.7f, 0.8f, 0.9f };
    settings.projection = 1;
    settings.fovY = 0.1f;
    settings.orthoScale = 0.2f;
    settings.exposure = 0.3f;
    settings.aperture = 0.4f;
    settings.focalDistance = 0.5f;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<CameraSettings_V1>();
    REQUIRE(settings.eye == settings2.eye);
    REQUIRE(settings.target == settings2.target);
    REQUIRE(settings.up == settings2.up);
    REQUIRE(settings.projection == settings2.projection);
    REQUIRE(settings.fovY == settings2.fovY);
    REQUIRE(settings.orthoScale == settings2.orthoScale);
    REQUIRE(settings.exposure == settings2.exposure);
    REQUIRE(settings.aperture == settings2.aperture);
    REQUIRE(settings.focalDistance == settings2.focalDistance);
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
    REQUIRE(settings.x == settings2.x);
    REQUIRE(settings.value == settings2.value);
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
    settings.mode = 1;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<LutParams_V1>();
    REQUIRE(settings.window == settings2.window);
    REQUIRE(settings.level == settings2.level);
    REQUIRE(settings.isovalue == settings2.isovalue);
    REQUIRE(settings.isorange == settings2.isorange);
    REQUIRE(settings.pctLow == settings2.pctLow);
    REQUIRE(settings.pctHigh == settings2.pctHigh);
    // REQUIRE(settings.controlPoints == settings2.controlPoints);
    REQUIRE(settings.mode == settings2.mode);
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
                           1 };
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<ChannelSettings_V1>();
    REQUIRE(settings.enabled == settings2.enabled);
    REQUIRE(settings.lutParams.window == settings2.lutParams.window);
    REQUIRE(settings.lutParams.level == settings2.lutParams.level);
    REQUIRE(settings.lutParams.isovalue == settings2.lutParams.isovalue);
    REQUIRE(settings.lutParams.isorange == settings2.lutParams.isorange);
    REQUIRE(settings.lutParams.pctLow == settings2.lutParams.pctLow);
    REQUIRE(settings.lutParams.pctHigh == settings2.lutParams.pctHigh);
    // REQUIRE(settings.lutParams.controlPoints == settings2.lutParams.controlPoints);
    REQUIRE(settings.lutParams.mode == settings2.lutParams.mode);
  }
}
