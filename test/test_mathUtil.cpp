#include <catch2/catch_test_macros.hpp>

#include "renderlib/MathUtil.h"

static constexpr float epsilon = 0.000001f;

TEST_CASE("Linear Space can be inverted", "[LinearSpace3f]")
{
  SECTION("Compute inverse of identity")
  {
    LinearSpace3f l;
    LinearSpace3f inv = l.inverse();
    REQUIRE(glm::all(glm::epsilonEqual(inv.vx, glm::vec3(1.0f, 0.0f, 0.0f), epsilon)));
    REQUIRE(glm::all(glm::epsilonEqual(inv.vy, glm::vec3(0.0f, 1.0f, 0.0f), epsilon)));
    REQUIRE(glm::all(glm::epsilonEqual(inv.vz, glm::vec3(0.0f, 0.0f, 1.0f), epsilon)));
  }

  SECTION("Inverse of rotation is transpose")
  {
    // make a rotated linear space
    LinearSpace3f l;
    l.vx = glm::vec3(0.0f, 0.0f, 1.0f);
    l.vy = glm::vec3(1.0f, 0.0f, 0.0f);
    l.vz = glm::vec3(0.0f, 1.0f, 0.0f);
    LinearSpace3f inv = l.inverse();
    REQUIRE(glm::all(glm::epsilonEqual(inv.vx, glm::vec3(0.0f, 1.0f, 0.0f), epsilon)));
    REQUIRE(glm::all(glm::epsilonEqual(inv.vy, glm::vec3(0.0f, 0.0f, 1.0f), epsilon)));
    REQUIRE(glm::all(glm::epsilonEqual(inv.vz, glm::vec3(1.0f, 0.0f, 0.0f), epsilon)));

    l.vx = glm::vec3(0.0f, 1.0f, 0.0f);
    l.vy = glm::vec3(0.0f, 0.0f, 1.0f);
    l.vz = glm::vec3(1.0f, 0.0f, 0.0f);
    inv = l.inverse();
    REQUIRE(glm::all(glm::epsilonEqual(inv.vx, glm::vec3(0.0f, 0.0f, 1.0f), epsilon)));
    REQUIRE(glm::all(glm::epsilonEqual(inv.vy, glm::vec3(1.0f, 0.0f, 0.0f), epsilon)));
    REQUIRE(glm::all(glm::epsilonEqual(inv.vz, glm::vec3(0.0f, 1.0f, 0.0f), epsilon)));

    // construct a LinearSpace from a glm rotation matrix of 3 Euler angles
    glm::mat3 m = glm::mat3_cast(glm::quat(glm::vec3(glm::radians(37.0f), glm::radians(55.0f), glm::radians(17.0f))));
    l = LinearSpace3f(m[0], m[1], m[2]);
    inv = l.inverse();
    auto mt = glm::transpose(m);
    REQUIRE(glm::all(glm::epsilonEqual(inv.vx, mt[0], epsilon)));
    REQUIRE(glm::all(glm::epsilonEqual(inv.vy, mt[1], epsilon)));
    REQUIRE(glm::all(glm::epsilonEqual(inv.vz, mt[2], epsilon)));
  }
}

TEST_CASE("Affine Space can be inverted", "[AffineSpace3f]")
{
  SECTION("Position is trivially inverted")
  {
    AffineSpace3f a;
    a.p = glm::vec3(1.0f, 2.0f, 3.0f);
    AffineSpace3f inv = a.inverse();
    REQUIRE(glm::all(glm::epsilonEqual(inv.p, glm::vec3(-1.0f, -2.0f, -3.0f), epsilon)));
  }
  SECTION("Position is properly inverted across rotations")
  {
    // make a rotated linear space
    AffineSpace3f a;
    a.l.vx = glm::vec3(0.0f, 0.0f, 1.0f);
    a.l.vy = glm::vec3(1.0f, 0.0f, 0.0f);
    a.l.vz = glm::vec3(0.0f, 1.0f, 0.0f);
    a.p = glm::vec3(1.0f, 2.0f, 3.0f);
    AffineSpace3f inv = a.inverse();
    REQUIRE(glm::all(glm::epsilonEqual(inv.p, glm::vec3(-3.0f, -1.0f, -2.0f), epsilon)));

    a.l.vx = glm::vec3(0.0f, 1.0f, 0.0f);
    a.l.vy = glm::vec3(0.0f, 0.0f, 1.0f);
    a.l.vz = glm::vec3(1.0f, 0.0f, 0.0f);
    inv = a.inverse();
    REQUIRE(glm::all(glm::epsilonEqual(inv.p, glm::vec3(-2.0f, -3.0f, -1.0f), epsilon)));
  }
  SECTION("Rotation is properly inverted across rotations")
  {
    // construct a LinearSpace from a glm rotation matrix of 3 Euler angles
    glm::mat3 m = glm::mat3_cast(glm::quat(glm::vec3(glm::radians(37.0f), glm::radians(55.0f), glm::radians(17.0f))));
    AffineSpace3f a;
    a.l = LinearSpace3f(m[0], m[1], m[2]);
    AffineSpace3f inv = a.inverse();

    // transform a vector through the original and then the inverse and get same vector.
    glm::vec3 testVec = glm::vec3(11.0f, 12.0f, 13.0f);
    glm::vec3 v1 = xfmPoint(a, testVec);
    glm::vec3 v2 = xfmPoint(inv, v1);
    REQUIRE(glm::all(glm::epsilonEqual(v2, testVec, epsilon)));
  }
}
