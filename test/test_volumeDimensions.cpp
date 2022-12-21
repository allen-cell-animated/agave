#include <catch2/catch_test_macros.hpp>

#include "renderlib/Logging.h"
#include "renderlib/VolumeDimensions.h"

TEST_CASE("VolumeDimensions", "[volumeDimensions]")
{
  // do not log errors from validation during testing
  Logging::Enable(false);
  SECTION("Default dims are valid")
  {
    VolumeDimensions d;
    bool ok = d.validate();
    REQUIRE(ok);
  }
  SECTION("Validate dimension orderings")
  {
    VolumeDimensions d;
    std::vector<std::string> good_orders = {
      "XYZCT", "XYZTC", "XYCTZ", "XYCZT", "XYTCZ", "XYTZC", "YXZCT", "YXZTC", "YXCTZ", "YXCZT", "YXTCZ", "YXTZC",
    };
    std::vector<std::string> bad_orders = { "TCZYX", "TZCYX", "CZTYX", "CTZYX", "ZCTYX", "ZTCYX", "TCZXY",  "TZCXY",
                                            "CZTXY", "CTZXY", "ZCTXY", "ZTCXY", "XX",    "ABC",   "XYCZTB", "XXZCT" };
    for (auto test : good_orders) {
      d.dimensionOrder = test;
      bool ok = d.validate();
      REQUIRE(ok);
    }
    for (auto test : bad_orders) {
      d.dimensionOrder = test;
      bool ok = d.validate();
      REQUIRE(!ok);
    }
  }
  SECTION("Validate plane indices")
  {
    VolumeDimensions d;
    d.sizeC = 6;
    d.sizeZ = 65;
    d.sizeT = 1;
    int testZ = 30;
    int testC = 1;
    int testT = 0;
    std::vector<std::pair<std::string, int>> tests = {
      { "XYZCT", 65 * 1 + 30 }, { "XYZTC", 65 * 1 + 30 }, { "XYCTZ", 6 * 30 + 1 },  { "XYCZT", 6 * 30 + 1 },
      { "XYTCZ", 6 * 30 + 1 },  { "XYTZC", 65 * 1 + 30 }, { "YXZCT", 65 * 1 + 30 }, { "YXZTC", 65 * 1 + 30 },
      { "YXCTZ", 6 * 30 + 1 },  { "YXCZT", 6 * 30 + 1 },  { "YXTCZ", 6 * 30 + 1 },  { "YXTZC", 65 * 1 + 30 }
    };
    for (auto test : tests) {
      d.dimensionOrder = test.first;
      int result = d.getPlaneIndex(testZ, testC, testT);
      REQUIRE(result == test.second);
    }
  }
  SECTION("zero-length dims are invalid")
  {
    {
      VolumeDimensions d;
      d.sizeX = 0;
      REQUIRE(!d.validate());
    }
    {
      VolumeDimensions d;
      d.sizeY = 0;
      REQUIRE(!d.validate());
    }
    {
      VolumeDimensions d;
      d.sizeZ = 0;
      REQUIRE(!d.validate());
    }
    {
      VolumeDimensions d;
      d.sizeC = 0;
      REQUIRE(!d.validate());
    }
    {
      VolumeDimensions d;
      d.sizeT = 0;
      REQUIRE(!d.validate());
    }
  }
  Logging::Enable(true);
}
