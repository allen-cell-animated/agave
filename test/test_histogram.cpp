#include <catch2/catch_test_macros.hpp>

#include "renderlib/Histogram.h"

TEST_CASE("Histogram edge cases are stable", "[histogram]")
{
  SECTION("Histogram of single value")
  {
    static const uint16_t VALUE = 257;
    uint16_t data[] = { VALUE, VALUE, VALUE };
    int COUNT = sizeof(data) / sizeof(data[0]);
    Histogram h(data, COUNT);

    REQUIRE(h._pixelCount == COUNT);
    REQUIRE(h._dataMax == VALUE);
    REQUIRE(h._dataMin == VALUE);

    // all data in first bin
    REQUIRE(h._ccounts[h._bins.size() - 1] == h._pixelCount);
    REQUIRE(h._ccounts[0] == h._pixelCount);
    REQUIRE(h._bins[0] == h._pixelCount);
  }

  SECTION("Histogram of binary segmentation")
  {
    static const uint16_t VALUE = 257;
    uint16_t data[] = { 0, 0, VALUE, VALUE };
    int COUNT = sizeof(data) / sizeof(data[0]);
    Histogram h(data, COUNT);

    REQUIRE(h._pixelCount == COUNT);
    REQUIRE(h._dataMax == VALUE);
    REQUIRE(h._dataMin == 0);

    // only 2 bins with data
    REQUIRE(h._bins[0] == 2);
    REQUIRE(h._bins[h._bins.size() - 1] == 2);
    REQUIRE(h._ccounts[h._bins.size() - 1] == h._pixelCount);
    REQUIRE(h._ccounts[h._bins.size() - 2] == 2);
    REQUIRE(h._ccounts[1] == 2);
    REQUIRE(h._ccounts[0] == 2);
  }

  SECTION("Histogram binning accuracy is good")
  {
    uint16_t data[] = { 0, 0, 1, 1, 2, 2, 510, 510, 511, 511, 512, 512 };
    int COUNT = sizeof(data) / sizeof(data[0]);
    Histogram h(data, COUNT);

    REQUIRE(h._pixelCount == COUNT);
    REQUIRE(h._dataMax == 512);
    REQUIRE(h._dataMin == 0);

    REQUIRE(h._bins[0] == 2);
    REQUIRE(h._bins[1] == 2);
    REQUIRE(h._bins[2] == 2);
    REQUIRE(h._bins[h._bins.size() / 2 - 1] == 0);
    REQUIRE(h._bins[h._bins.size() - 2] == 2);
    REQUIRE(h._bins[h._bins.size() - 1] == 2);
    REQUIRE(h._ccounts[h._bins.size() - 1] == h._pixelCount);
    REQUIRE(h._ccounts[h._bins.size() - 2] == h._pixelCount - 2);
    REQUIRE(h._ccounts[1] == 4);
    REQUIRE(h._ccounts[0] == 2);
  }
}

TEST_CASE("Histogram LUT generation is working", "[histogram]")
{
  SECTION("Simple linear gradient is working")
  {
    Histogram h(nullptr, 0);

    // simple straight line
    std::vector<LutControlPoint> pts = { { 0.0f, 0.0f }, { 1.0f, 1.0f } };
    float* lut = h.generate_controlPoints(pts);

    REQUIRE(lut[0] == 0.0);
    REQUIRE(lut[127] < 0.5);
    REQUIRE(lut[128] > 0.5);
    REQUIRE(lut[255] == 1.0);
  }

  SECTION("Step function control points generate a proper Lut")
  {
    Histogram h(nullptr, 0);

    // discontinuity at 0.5
    std::vector<LutControlPoint> pts = { { 0.0f, 0.0f }, { 0.5f, 0.0f }, { 0.5f, 1.0f }, { 1.0f, 1.0f } };
    float* lut = h.generate_controlPoints(pts);

    REQUIRE(lut[0] == 0.0);
    REQUIRE(lut[127] == 0.0);
    REQUIRE(lut[128] == 1.0);
    REQUIRE(lut[255] == 1.0);
  }

  SECTION("Non-degenerate Step function control points generate a proper Lut")
  {
    Histogram h(nullptr, 0);

    // discontinuity at 0.5
    std::vector<LutControlPoint> pts = { { 0.0f, 0.0f }, { 0.49999f, 0.0f }, { 0.500001f, 1.0f }, { 1.0f, 1.0f } };
    float* lut = h.generate_controlPoints(pts);

    REQUIRE(lut[0] == 0.0);
    REQUIRE(lut[127] == 0.0);
    REQUIRE(lut[128] == 1.0);
    REQUIRE(lut[255] == 1.0);
  }

  SECTION("Control points starting at non-zero will fill in")
  {
    Histogram h(nullptr, 0);

    // discontinuity at 0.5
    std::vector<LutControlPoint> pts = { { 0.5f, 0.5f }, { 1.0f, 1.0f } };
    float* lut = h.generate_controlPoints(pts);

    REQUIRE(lut[0] == 0.0);
    REQUIRE(lut[1] == 0.0);
    REQUIRE(lut[127] == 0.0);
    REQUIRE(lut[128] > 0.5);
    REQUIRE(lut[255] == 1.0);
  }

  SECTION("Control points aligned with lut array values are exact")
  {
    Histogram h(nullptr, 0);

    // value at 16/255, and at 197/255
    std::vector<LutControlPoint> pts = {
      { 0.0f, 0.0f }, { 16.0f / 255.0f, 0.5f }, { 197.0f / 255.0f, 0.75f }, { 1.0f, 1.0f }
    };
    float* lut = h.generate_controlPoints(pts);

    REQUIRE(lut[16] == 0.5);
    REQUIRE(lut[197] == 0.75);
  }
}
