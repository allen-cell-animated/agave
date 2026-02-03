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
    // REQUIRE(h._ccounts[h._bins.size() - 1] == h.getPixelCount());
    // REQUIRE(h._ccounts[0] == h.getPixelCount());
    REQUIRE(h.getBinCount(0) == h.getPixelCount());
  }

  SECTION("Histogram of binary segmentation")
  {
    static const uint16_t VALUE = 257;
    uint16_t data[] = { 0, 0, VALUE, VALUE };
    int COUNT = sizeof(data) / sizeof(data[0]);
    Histogram h(data, COUNT, 512, false); // disable outlier filtering for label data

    REQUIRE(h.getPixelCount() == COUNT);
    REQUIRE(h.getDataMax() == VALUE);
    REQUIRE(h.getDataMin() == 0);

    // only 2 bins with data
    REQUIRE(h.getBinCount(0) == 2);
    REQUIRE(h.getBinCount(h.getNumBins() - 1) == 2);
    // REQUIRE(h._ccounts[h._bins.size() - 1] == h.getPixelCount());
    // REQUIRE(h._ccounts[h._bins.size() - 2] == 2);
    // REQUIRE(h._ccounts[1] == 2);
    // REQUIRE(h._ccounts[0] == 2);
  }

  SECTION("Histogram binning accuracy is good")
  {
    uint16_t data[] = { 0, 0, 1, 1, 2, 2, 510, 510, 511, 511, 512, 512 };
    int COUNT = sizeof(data) / sizeof(data[0]);
    Histogram h(data, COUNT, 512, false); // disable outlier filtering for precise binning test

    REQUIRE(h.getPixelCount() == COUNT);
    REQUIRE(h.getDataMax() == 512);
    REQUIRE(h.getDataMin() == 0);

    REQUIRE(h.getBinCount(0) == 2);
    REQUIRE(h.getBinCount(1) == 2);
    REQUIRE(h.getBinCount(2) == 2);
    REQUIRE(h.getBinCount(h.getNumBins() / 2 - 1) == 0);
    REQUIRE(h.getBinCount(h.getNumBins() - 2) == 2);
    REQUIRE(h.getBinCount(h.getNumBins() - 1) == 2);
    // REQUIRE(h._ccounts[h._bins.size() - 1] == h.getPixelCount());
    // REQUIRE(h._ccounts[h._bins.size() - 2] == h.getPixelCount() - 2);
    // REQUIRE(h._ccounts[1] == 4);
    // REQUIRE(h._ccounts[0] == 2);
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

    // With outlier filtering (default)
    Histogram hFiltered(data.data(), data.size(), 512, true);

    // Without outlier filtering
    Histogram hUnfiltered(data.data(), data.size(), 512, false);

    // Unfiltered should use full range
    REQUIRE(hUnfiltered._dataMin == 0);
    REQUIRE(hUnfiltered._dataMax == 65535);

    // Filtered should exclude the outliers
    REQUIRE(hFiltered._dataMin > 0);
    REQUIRE(hFiltered._dataMax < 65535);
    REQUIRE(hFiltered._dataMin >= 100);
    REQUIRE(hFiltered._dataMax <= 200);
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
    Histogram h(data.data(), data.size(), 512, false);

    REQUIRE(h._dataMin == 0);
    REQUIRE(h._dataMax == 4);
    REQUIRE(h._pixelCount == 1000);
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

    Histogram hFiltered(data.data(), data.size(), 512, true);
    Histogram hUnfiltered(data.data(), data.size(), 512, false);

    // Unfiltered includes outliers
    REQUIRE(hUnfiltered._dataMin == 0);
    REQUIRE(hUnfiltered._dataMax == 10000);

    // Filtered excludes outliers
    REQUIRE(hFiltered._dataMin >= 950);
    REQUIRE(hFiltered._dataMax <= 1050);
  }

  SECTION("Outlier filtering handles uniform distribution correctly")
  {
    // Uniform distribution from 1000 to 2000
    std::vector<uint16_t> data;
    for (int i = 1000; i <= 2000; ++i) {
      data.push_back(i);
    }

    Histogram h(data.data(), data.size(), 512, true);

    // Should preserve most of the range (maybe trim very edges)
    REQUIRE(h._dataMin >= 1000);
    REQUIRE(h._dataMin <= 1010);
    REQUIRE(h._dataMax >= 1990);
    REQUIRE(h._dataMax <= 2000);
  }

  SECTION("Outlier filtering is stable with all identical values")
  {
    uint16_t data[] = { 500, 500, 500, 500, 500 };
    int COUNT = sizeof(data) / sizeof(data[0]);

    Histogram hFiltered(data, COUNT, 512, true);
    Histogram hUnfiltered(data, COUNT, 512, false);

    // Both should handle identical values the same way
    REQUIRE(hFiltered._dataMin == 500);
    REQUIRE(hFiltered._dataMax == 500);
    REQUIRE(hUnfiltered._dataMin == 500);
    REQUIRE(hUnfiltered._dataMax == 500);
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

    Histogram hFiltered(data.data(), data.size(), 512, true);

    // Should exclude the extreme outliers at 10 and 50000
    REQUIRE(hFiltered._dataMin >= 1000);
    REQUIRE(hFiltered._dataMax < 50000);
  }

  SECTION("Percentile constants are accessible")
  {
    // Verify the constants are defined and reasonable
    REQUIRE(Histogram::HISTOGRAM_RANGE_PCT_LOW > 0.0f);
    REQUIRE(Histogram::HISTOGRAM_RANGE_PCT_LOW < 0.01f);  // Less than 1%
    REQUIRE(Histogram::HISTOGRAM_RANGE_PCT_HIGH > 0.99f); // Greater than 99%
    REQUIRE(Histogram::HISTOGRAM_RANGE_PCT_HIGH < 1.0f);
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

    Histogram h(data.data(), data.size(), 512, true);

    // Total pixel count should still be correct
    REQUIRE(h._pixelCount == 1002);

    // Cumulative counts should add up to total pixels
    REQUIRE(h._ccounts[h._ccounts.size() - 1] == h._pixelCount);

    // Outliers should be clamped to boundary bins
    REQUIRE(h._bins[0] > 0);                  // Low outlier clamped to first bin
    REQUIRE(h._bins[h._bins.size() - 1] > 0); // High outlier clamped to last bin
  }
}

TEST_CASE("Histogram bin_range calculation uses robust range", "[histogram]")
{
  SECTION("bin_range uses filtered data range, not absolute range")
  {
    // Data with outliers
    std::vector<uint16_t> data;
    data.push_back(0); // outlier
    for (int i = 0; i < 1000; ++i) {
      data.push_back(1000 + i % 100); // 1000-1099
    }
    data.push_back(65535); // outlier

    Histogram hFiltered(data.data(), data.size(), 512, true);

    float firstBinCenter, lastBinCenter, binSize;
    hFiltered.bin_range(512, firstBinCenter, lastBinCenter, binSize);

    // Bin range should correspond to filtered range, not 0-65535
    REQUIRE(firstBinCenter >= 1000.0f);
    REQUIRE(firstBinCenter < 1100.0f);
    REQUIRE(lastBinCenter >= 1000.0f);
    REQUIRE(lastBinCenter < 1200.0f);

    // Bin size should be much smaller than if full range was used
    REQUIRE(binSize < 1.0f); // With full range, binSize would be ~128
  }
}
