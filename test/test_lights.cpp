#include <catch2/catch_test_macros.hpp>

#include "renderlib/Light.h"
#include "renderlib/SceneLight.h"

static constexpr float kEpsilon = 1e-4f;
static constexpr float kAngleEpsilon = 1e-4f;

static void
requireOrthonormal(const Light& light)
{
  REQUIRE(glm::epsilonEqual(glm::length(light.m_N), 1.0f, kEpsilon));
  REQUIRE(glm::epsilonEqual(glm::length(light.m_U), 1.0f, kEpsilon));
  REQUIRE(glm::epsilonEqual(glm::length(light.m_V), 1.0f, kEpsilon));
  REQUIRE(glm::epsilonEqual(glm::dot(light.m_N, light.m_U), 0.0f, kEpsilon));
  REQUIRE(glm::epsilonEqual(glm::dot(light.m_N, light.m_V), 0.0f, kEpsilon));
  REQUIRE(glm::epsilonEqual(glm::dot(light.m_U, light.m_V), 0.0f, kEpsilon));
}

static void
requireAngleConsistency(const Light& light, const glm::vec3& expectedDir)
{
  glm::vec3 dirFromAngles(0.0f);
  Light::sphericalToCartesian(light.m_Phi, light.m_Theta, dirFromAngles);
  dirFromAngles = glm::normalize(dirFromAngles);

  const glm::vec3 normExpected = glm::normalize(expectedDir);
  REQUIRE(glm::epsilonEqual(glm::dot(dirFromAngles, normExpected), 1.0f, kAngleEpsilon));
}

static void
requireLightGeometry(const Light& light, const glm::vec3& expectedDir)
{
  if (light.m_T == LightType_Sphere) {
    const glm::vec3 expectedP = light.m_Target + glm::normalize(expectedDir);
    REQUIRE(glm::all(glm::epsilonEqual(light.m_P, expectedP, kEpsilon)));
    REQUIRE(glm::epsilonEqual(glm::dot(light.m_N, glm::normalize(expectedDir)), 1.0f, kEpsilon));
  } else {
    const glm::vec3 expectedP = light.m_Target + light.m_Distance * glm::normalize(expectedDir);
    REQUIRE(glm::all(glm::epsilonEqual(light.m_P, expectedP, kEpsilon)));
    REQUIRE(glm::epsilonEqual(glm::dot(light.m_N, -glm::normalize(expectedDir)), 1.0f, kEpsilon));
  }
  requireOrthonormal(light);
  requireAngleConsistency(light, expectedDir);
}

TEST_CASE("SceneLight updateTransform preserves basis", "[Light]")
{
  SECTION("Area light keeps distance and basis")
  {
    Light light;
    light.m_T = LightType_Area;
    light.m_Target = glm::vec3(1.0f, 2.0f, 3.0f);
    light.m_Distance = 4.0f;
    light.m_U = glm::vec3(0.0f, 1.0f, 0.0f);

    SceneLight sceneLight(&light);
    sceneLight.m_transform.m_rotation = glm::quat(glm::vec3(glm::radians(35.0f), glm::radians(20.0f), 0.0f));
    sceneLight.updateTransform();

    REQUIRE(glm::epsilonEqual(glm::length(light.m_P - light.m_Target), light.m_Distance, kEpsilon));
    requireOrthonormal(light);
  }

  SECTION("Sphere light preserves roll even if U is degenerate")
  {
    Light light;
    light.m_T = LightType_Sphere;
    light.m_Target = glm::vec3(0.0f);
    light.m_U = glm::vec3(0.0f, 0.0f, 1.0f);

    SceneLight sceneLight(&light);
    sceneLight.m_transform.m_rotation = glm::quat(glm::vec3(glm::radians(80.0f), 0.0f, 0.0f));
    sceneLight.updateTransform();

    requireOrthonormal(light);
  }
}

TEST_CASE("Light orientation consistency across rotations", "[Light]")
{
  struct RotationCase
  {
    const char* name;
    glm::quat rotation;
  };

  const RotationCase rotations[] = {
    { "identity", glm::quat(glm::vec3(0.0f, 0.0f, 0.0f)) },
    { "yaw", glm::quat(glm::vec3(0.0f, glm::radians(45.0f), 0.0f)) },
    { "pitch", glm::quat(glm::vec3(glm::radians(35.0f), 0.0f, 0.0f)) },
    { "roll", glm::quat(glm::vec3(0.0f, 0.0f, glm::radians(60.0f))) },
    { "combined", glm::quat(glm::vec3(glm::radians(20.0f), glm::radians(35.0f), glm::radians(15.0f))) },
  };

  for (const auto& test : rotations) {
    SECTION(test.name)
    {
      const glm::vec3 target(1.0f, -2.0f, 0.5f);
      const float distance = 3.5f;
      const glm::vec3 expectedDir = test.rotation * glm::vec3(0.0f, 0.0f, 1.0f);

      Light areaLight;
      areaLight.m_T = LightType_Area;
      areaLight.m_Target = target;
      areaLight.m_Distance = distance;

      SceneLight areaSceneLight(&areaLight);
      areaSceneLight.m_transform.m_rotation = test.rotation;
      areaSceneLight.updateTransform();
      requireLightGeometry(areaLight, expectedDir);

      Light sphereLight;
      sphereLight.m_T = LightType_Sphere;
      sphereLight.m_Target = target;

      SceneLight sphereSceneLight(&sphereLight);
      sphereSceneLight.m_transform.m_rotation = test.rotation;
      sphereSceneLight.updateTransform();
      requireLightGeometry(sphereLight, expectedDir);
    }
  }
}
