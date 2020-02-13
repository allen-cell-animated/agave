#include "catch.hpp"

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
      "XYZCT", "XYZTC", "XYCTZ", "XYCTZ", "XYTCZ", "XYTZC", "YXZCT", "YXZTC", "YXCTZ", "YXCTZ", "YXTCZ", "YXTZC",
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
