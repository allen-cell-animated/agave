#include <catch2/catch_test_macros.hpp>

#include "../agave_app/Serialize.h"
#include "renderlib/json/json.hpp"

TEST_CASE("Json Serialization", "[serialize]")
{
  SECTION("Read and write json defaults")
  {
    auto settings = PathTraceSettings_V1{};
    nlohmann::json json = settings;
    auto settings2 = json.get<PathTraceSettings_V1>();
    REQUIRE(settings.primaryStepSize == settings2.primaryStepSize);
    REQUIRE(settings.secondaryStepSize == settings2.secondaryStepSize);
  }
}
