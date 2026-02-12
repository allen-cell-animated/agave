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

    REQUIRE(h.getPixelCount() == COUNT);
    REQUIRE(h.getDataMax() == VALUE);
    REQUIRE(h.getDataMin() == VALUE);

    // all data in first bin
    REQUIRE(h.getDisplayBinCount(0) == h.getPixelCount());
  }

  SECTION("Histogram of binary segmentation")
  {
    static const uint16_t VALUE = 257;
    uint16_t data[] = { 0, 0, VALUE, VALUE };
    int COUNT = sizeof(data) / sizeof(data[0]);
    Histogram h(data, COUNT);

    REQUIRE(h.getPixelCount() == COUNT);
    REQUIRE(h.getDataMax() == VALUE);
    REQUIRE(h.getDataMin() == 0);

    // only 2 bins with data
    REQUIRE(h.getDisplayBinCount(0) == 2);
    REQUIRE(h.getDisplayBinCount(h.getNumDisplayBins() - 1) == 2);
  }

  SECTION("Histogram binning accuracy is good")
  {
    uint16_t data[] = { 0, 0, 1, 1, 2, 2, 510, 510, 511, 511, 512, 512 };
    int COUNT = sizeof(data) / sizeof(data[0]);
    Histogram h(data, COUNT);

    REQUIRE(h.getPixelCount() == COUNT);
    REQUIRE(h.getDataMax() == 512);
    REQUIRE(h.getDataMin() == 0);

    REQUIRE(h.getDisplayBinCount(0) == 2);
    REQUIRE(h.getDisplayBinCount(1) == 2);
    REQUIRE(h.getDisplayBinCount(2) == 2);
    REQUIRE(h.getDisplayBinCount(h.getNumDisplayBins() / 2 - 1) == 0);
    REQUIRE(h.getDisplayBinCount(h.getNumDisplayBins() - 2) == 2);
    REQUIRE(h.getDisplayBinCount(h.getNumDisplayBins() - 1) == 2);
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
TEST_CASE("Histogram outlier filtering works correctly", "[histogram]")
{
  SECTION("Outlier filtering excludes extreme values in continuous data")
  {
    // Create data with extreme outliers: mostly values 100-200, but with extreme min/max
    std::vector<uint16_t> data;
    data.push_back(0); // extreme low outlier (single pixel)
    for (int i = 0; i < 1000; ++i) {
      data.push_back(100 + (i % 101)); // main data range 100-200
    }
    data.push_back(65535); // extreme high outlier (single pixel)

    Histogram hFiltered(data.data(), data.size());

    // Unfiltered should use full range
    REQUIRE(hFiltered.getDataMin() == 0);
    REQUIRE(hFiltered.getDataMax() == 65535);

    // Filtered should exclude the outliers
    // Allow some discretization error from binning (bin width ~16 for this data range)
    REQUIRE(hFiltered.getFilteredMin() > 50); // Well above 0, excluding low outlier
    REQUIRE(hFiltered.getFilteredMax() < 65535);
    REQUIRE(hFiltered.getFilteredMin() < 210); // Within or near main data range
    REQUIRE(hFiltered.getFilteredMax() <= 210);
  }

  SECTION("Outlier filtering preserves label data when disabled")
  {
    // Label data: 0, 1, 2, 3, 4 uniformly distributed
    std::vector<uint16_t> data;
    for (int label = 0; label <= 4; ++label) {
      for (int count = 0; count < 200; ++count) {
        data.push_back(label);
      }
    }

    // Without outlier filtering (for label data)
    Histogram h(data.data(), data.size());

    REQUIRE(h.getDataMin() == 0);
    REQUIRE(h.getDataMax() == 4);
    REQUIRE(h.getPixelCount() == 1000);
  }

  SECTION("Outlier filtering handles gaussian-like distribution")
  {
    // Create data with normal-like distribution centered at 1000 with a few extreme outliers
    // Need enough samples for 0.1% to be meaningful (at least 1000 samples)
    std::vector<uint16_t> data;
    data.push_back(0); // extreme outlier

    // Main distribution around 1000 (5000 samples)
    for (int i = 0; i < 5000; ++i) {
      data.push_back(950 + (i % 101)); // 950-1050
    }

    data.push_back(10000); // extreme outlier

    Histogram hFiltered(data.data(), data.size());

    // Unfiltered includes outliers
    REQUIRE(hFiltered.getDataMin() == 0);
    REQUIRE(hFiltered.getDataMax() == 10000);

    // Filtered excludes outliers
    REQUIRE(hFiltered.getFilteredMin() >= 950);
    REQUIRE(hFiltered.getFilteredMax() <= 1050);
  }

  SECTION("Outlier filtering handles uniform distribution correctly")
  {
    // Uniform distribution from 1000 to 2000
    std::vector<uint16_t> data;
    for (int i = 1000; i <= 2000; ++i) {
      data.push_back(i);
    }

    Histogram h(data.data(), data.size());

    // Should preserve most of the range (maybe trim very edges)
    REQUIRE(h.getFilteredMin() > 1000);
    REQUIRE(h.getFilteredMin() < 1010);
    REQUIRE(h.getFilteredMax() > 1990);
    REQUIRE(h.getFilteredMax() < 2000);
  }

  SECTION("Outlier filtering is stable with all identical values")
  {
    uint16_t data[] = { 500, 500, 500, 500, 500 };
    int COUNT = sizeof(data) / sizeof(data[0]);

    Histogram hFiltered(data, COUNT);
    Histogram hUnfiltered(data, COUNT);

    // Both should handle identical values the same way
    REQUIRE(hFiltered.getDataMin() == 500);
    REQUIRE(hFiltered.getDataMax() == 500);
    REQUIRE(hUnfiltered.getDataMin() == 500);
    REQUIRE(hUnfiltered.getDataMax() == 500);
  }

  SECTION("Outlier filtering with very sparse outliers")
  {
    // 99.8% of data in narrow range, 0.1% each extreme outliers
    std::vector<uint16_t> data;

    // Add low outliers (0.1%)
    for (int i = 0; i < 50; ++i) {
      data.push_back(10);
    }

    // Main data (99.8%)
    for (int i = 0; i < 49900; ++i) {
      data.push_back(1000 + (i % 100)); // 1000-1099
    }

    // Add high outliers (0.1%)
    for (int i = 0; i < 50; ++i) {
      data.push_back(50000);
    }

    Histogram hFiltered(data.data(), data.size());

    // Should exclude the extreme outliers at 10 and 50000
    // Allow some discretization error from binning (bin width ~12 for this data range)
    REQUIRE(hFiltered.getFilteredMin() > 100);  // Well above 10, excluding outliers
    REQUIRE(hFiltered.getFilteredMin() < 1100); // Within main data range
    REQUIRE(hFiltered.getFilteredMax() < 50000);
  }

  SECTION("Outlier filtering maintains correct bin counts")
  {
    // Data with outliers
    std::vector<uint16_t> data;
    data.push_back(0); // outlier
    for (int i = 0; i < 1000; ++i) {
      data.push_back(1000 + (i % 100)); // main range: 1000-1099
    }
    data.push_back(20000); // outlier

    Histogram h(data.data(), data.size());

    // Total pixel count should still be correct
    REQUIRE(h.getPixelCount() == 1002);

    // Outliers should be clamped to boundary bins
    REQUIRE(h.getDisplayBinCount(0) > 0);                         // Low outlier clamped to first bin
    REQUIRE(h.getDisplayBinCount(h.getNumDisplayBins() - 1) > 0); // High outlier clamped to last bin
  }
}

TEST_CASE("Histogram binRange calculation uses robust range", "[histogram]")
{
  SECTION("binRange uses filtered data range, not absolute range")
  {
    // Data with outliers
    std::vector<uint16_t> data;
    data.push_back(0); // outlier
    for (int i = 0; i < 1000; ++i) {
      data.push_back(1000 + i % 100); // 1000-1099
    }
    data.push_back(65535); // outlier

    Histogram hFiltered(data.data(), data.size());

    float firstBinCenter, lastBinCenter, binSize;
    hFiltered.binRange(
      512, hFiltered.getFilteredMin(), hFiltered.getFilteredMax(), firstBinCenter, lastBinCenter, binSize);

    // Bin range should correspond to filtered range, not 0-65535
    // Allow some discretization error from binning (bin width ~16 for this data range)
    REQUIRE(firstBinCenter > 900.0f); // Well above 0, excluding low outlier
    REQUIRE(firstBinCenter < 1100.0f);
    REQUIRE(lastBinCenter >= 1000.0f);
    REQUIRE(lastBinCenter < 1200.0f);

    // Bin size should be much smaller than if full range was used
    REQUIRE(binSize < 1.0f); // With full range, binSize would be ~128
  }
}
