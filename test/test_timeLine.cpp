#include "catch.hpp"

#include "renderlib/Timeline.h"

TEST_CASE("Timeline", "[timeLine]")
{
  SECTION("Timeline increment/decrement wrapping works")
  {
    Timeline t(0, 3);

    REQUIRE(t.currentTime() == 0);
    t.increment(1);
    REQUIRE(t.currentTime() == 1);
    t.increment(-2);
    REQUIRE(t.currentTime() == 3);
    t.increment(3);
    REQUIRE(t.currentTime() == 2);
    t.increment(10);
    REQUIRE(t.currentTime() == 0);
    t.increment(-10);
    REQUIRE(t.currentTime() == 2);
  }
  SECTION("Timeline increment/decrement clamping works")
  {
    Timeline t(0, 3, false);

    REQUIRE(t.currentTime() == 0);
    t.increment(1);
    REQUIRE(t.currentTime() == 1);
    t.increment(-2);
    REQUIRE(t.currentTime() == 0);
    t.increment(3);
    REQUIRE(t.currentTime() == 3);
    t.increment(10);
    REQUIRE(t.currentTime() == 3);
    t.increment(-10);
    REQUIRE(t.currentTime() == 0);
  }
  SECTION("Trivial default timeline works")
  {
    Timeline t;

    REQUIRE(t.currentTime() == 0);
    t.increment(1);
    REQUIRE(t.currentTime() == 0);
    t.increment(-2);
    REQUIRE(t.currentTime() == 0);
    t.increment(3);
    REQUIRE(t.currentTime() == 0);
    t.increment(10);
    REQUIRE(t.currentTime() == 0);
    t.increment(-10);
    REQUIRE(t.currentTime() == 0);
  }
}
