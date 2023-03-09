#include <catch2/catch_test_macros.hpp>

#include "renderlib/version.hpp"

TEST_CASE("Version class", "[version]")
{
  SECTION("Current version is consistent")
  {
    REQUIRE(CURRENT_VERSION.major() == AICS_VERSION_MAJOR);
    REQUIRE(CURRENT_VERSION.minor() == AICS_VERSION_MINOR);
    REQUIRE(CURRENT_VERSION.patch() == AICS_VERSION_PATCH);
  }

  SECTION("Version comparisons work")
  {
    REQUIRE(Version(1, 0, 0) < Version(1, 0, 1));
    REQUIRE(Version(1, 0, 0) <= Version(1, 0, 1));

    REQUIRE(Version(1, 2, 3) == Version(1, 2, 3));
    REQUIRE(Version(1, 2, 3) >= Version(1, 2, 3));
    REQUIRE(Version(1, 2, 3) <= Version(1, 2, 3));
    REQUIRE(!(Version(1, 2, 3) < Version(1, 2, 3)));
    REQUIRE(!(Version(1, 2, 3) > Version(1, 2, 3)));

    REQUIRE(!(Version(1, 2, 3) == Version(1, 2, 4)));
    REQUIRE((Version(1, 2, 3) != Version(1, 2, 4)));

    REQUIRE(Version(1, 2, 3) < Version(2, 2, 3));
    REQUIRE(Version(1, 2, 3) < Version(1, 3, 3));
    REQUIRE(Version(1, 2, 3) < Version(1, 2, 4));
  }
}
