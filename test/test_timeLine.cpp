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
    Timeline t(0, 3, Timeline::WrapMode::TIMELINE_CLAMP);

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
  SECTION("Changing the timeline range and clamp mode works")
  {
    Timeline t;

    REQUIRE(t.currentTime() == 0);
    t.increment(1);
    REQUIRE(t.currentTime() == 0);
    t.setRange(0, 7);
    REQUIRE(t.currentTime() == 0);
    t.increment(6);
    REQUIRE(t.currentTime() == 6);
    t.setRange(0, 4);
    REQUIRE(t.currentTime() == 1);

    // set to clamped
    t.setWrap(Timeline::WrapMode::TIMELINE_CLAMP);
    t.increment(7);
    REQUIRE(t.currentTime() == 4);
    t.setRange(0, 3);
    REQUIRE(t.currentTime() == 3);
  }
}
