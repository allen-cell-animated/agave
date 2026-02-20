#include <catch2/catch_test_macros.hpp>

#include "renderlib/Light.h"
#include "renderlib/SceneLight.h"

static constexpr float kEpsilon = 1e-4f;

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
