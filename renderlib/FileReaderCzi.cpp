#include "FileReaderCzi.h"

#include "BoundingBox.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "VolumeDimensions.h"

#include <libCZI/Src/libCZI/libCZI.h>

#include "pugixml/pugixml.hpp"

#include <boost/filesystem.hpp>

#include <chrono>
#include <map>
#include <set>

static const int IN_MEMORY_BPP = 16;

FileReaderCzi::FileReaderCzi() {}

FileReaderCzi::~FileReaderCzi() {}

class ScopedCziReader
{
public:
  ScopedCziReader(const std::string& filepath)
  {
    boost::filesystem::path fpath(filepath);
    const std::wstring widestr = fpath.wstring();

    std::shared_ptr<libCZI::IStream> stream = libCZI::CreateStreamFromFile(widestr.c_str());
    m_reader = libCZI::CreateCZIReader();

    m_reader->Open(stream);
  }
  ~ScopedCziReader()
  {
    if (m_reader) {
      m_reader->Close();
    }
  }
  std::shared_ptr<libCZI::ICZIReader> reader() { return m_reader; }

protected:
  std::shared_ptr<libCZI::ICZIReader> m_reader;
};

libCZI::IntRect
getSceneYXSize(libCZI::SubBlockStatistics& statistics, int sceneIndex = 0)
{
  bool hasScene = statistics.dimBounds.IsValid(libCZI::DimensionIndex::S);
  if (hasScene) {
    int sStart(0), sSize(0);
    statistics.dimBounds.TryGetInterval(libCZI::DimensionIndex::S, &sStart, &sSize);
    if (sceneIndex >= sStart && (sStart + sSize - 1) >= sceneIndex && !statistics.sceneBoundingBoxes.empty()) {
      return statistics.sceneBoundingBoxes[sceneIndex].boundingBoxLayer0;
    }
  } else {
    return statistics.boundingBoxLayer0Only;
  }
  return statistics.boundingBoxLayer0Only;
}

pugi_agave::xml_node
firstChild(pugi_agave::xml_node& el, std::string tag)
{
  pugi_agave::xml_node child = el.child(tag.c_str());
  if (!child) {
    LOG_ERROR << "No " << tag << "element in xml";
  }
  return child;
}

bool
readCziDimensions(const std::shared_ptr<libCZI::ICZIReader>& reader,
                  const std::string filepath,
                  libCZI::SubBlockStatistics& statistics,
                  VolumeDimensions& dims,
                  uint32_t scene)
{
  // metadata xml
  auto mds = reader->ReadMetadataSegment();
  std::shared_ptr<libCZI::ICziMetadata> md = mds->CreateMetaFromMetadataSegment();
  std::shared_ptr<libCZI::ICziMultiDimensionDocumentInfo> docinfo = md->GetDocumentInfo();

  libCZI::ScalingInfo scalingInfo = docinfo->GetScalingInfo();
  // convert meters to microns?
  dims.physicalSizeX = (float)(scalingInfo.scaleX * 1000000.0);
  dims.physicalSizeY = (float)(scalingInfo.scaleY * 1000000.0);
  dims.physicalSizeZ = (float)(scalingInfo.scaleZ * 1000000.0);

  // get all dimension bounds and enumerate.
  // I am making an assumption here that each scene has the same Z C and T dimensions.
  // If acquisition ended early, then the data might not be complete and that will have to be caught later.
  statistics.dimBounds.EnumValidDimensions([&](libCZI::DimensionIndex dimensionIndex, int start, int size) -> bool {
    switch (dimensionIndex) {
      case libCZI::DimensionIndex::Z:
        dims.sizeZ = size;
        break;
      case libCZI::DimensionIndex::C:
        dims.sizeC = size;
        break;
      case libCZI::DimensionIndex::T:
        dims.sizeT = size;
        break;
      default:
        break;
    }
    return true;
  });

  libCZI::IntRect planebox = getSceneYXSize(statistics, scene);
  dims.sizeX = planebox.w;
  dims.sizeY = planebox.h;

  std::string xml = md->GetXml();

  pugi_agave::xml_document czixml;
  pugi_agave::xml_parse_result parseOk = czixml.load_string(xml.c_str());
  if (!parseOk) {
    LOG_ERROR << "Bad CZI xml metadata content";
    return false;
  }

  // first Metadata child by tag
  pugi_agave::xml_node metadataEl = czixml.child("Metadata");
  if (!metadataEl) {
    LOG_ERROR << "No Metadata element in czi xml";
    return false;
  }
  pugi_agave::xml_node informationEl = firstChild(metadataEl, "Information");
  if (!informationEl) {
    return false;
  }
  pugi_agave::xml_node imageEl = firstChild(informationEl, "Image");
  if (!imageEl) {
    return false;
  }
  pugi_agave::xml_node dimensionsEl = firstChild(imageEl, "Dimensions");
  if (!dimensionsEl) {
    return false;
  }
  pugi_agave::xml_node channelsEl = firstChild(dimensionsEl, "Channels");
  if (!channelsEl) {
    return false;
  }
  std::vector<std::string> channelNames;
  for (pugi_agave::xml_node node : channelsEl.children("Channel")) {
    channelNames.push_back(node.attribute("Name").value());
  }
  dims.channelNames = channelNames;

  libCZI::SubBlockInfo info;
  bool ok = reader->TryGetSubBlockInfoOfArbitrarySubBlockInChannel(0, info);
  if (ok) {
    switch (info.pixelType) {
      case libCZI::PixelType::Gray8:
        dims.bitsPerPixel = 8;
        break;
      case libCZI::PixelType::Gray16:
        dims.bitsPerPixel = 16;
        break;
      case libCZI::PixelType::Gray32Float:
        dims.bitsPerPixel = 32;
        break;
      case libCZI::PixelType::Bgr24:
        dims.bitsPerPixel = 24;
        break;
      case libCZI::PixelType::Bgr48:
        dims.bitsPerPixel = 48;
        break;
      case libCZI::PixelType::Bgr96Float:
        dims.bitsPerPixel = 96;
        break;
      default:
        dims.bitsPerPixel = 0;
        return false;
    }
  } else {
    return false;
  }

  return dims.validate();
}

// DANGER: assumes dataPtr has enough space allocated!!!!
bool
readCziPlane(const std::shared_ptr<libCZI::ICZIReader>& reader,
             const libCZI::IntRect& planeRect,
             const libCZI::CDimCoordinate& planeCoord,
             const VolumeDimensions& volumeDims,
             uint8_t* dataPtr)
{
  auto accessor = reader->CreateSingleChannelTileAccessor();
  auto bitmap = accessor->Get(planeRect, &planeCoord, nullptr);
  libCZI::IntSize size = bitmap->GetSize();
  {
    libCZI::ScopedBitmapLockerSP lckScoped{ bitmap };
    assert(lckScoped.ptrDataRoi == lckScoped.ptrData);
    assert(volumeDims.sizeX == size.w);
    assert(volumeDims.sizeY == size.h);
    size_t bytesPerRow = size.w * 2; // destination stride
    if (volumeDims.bitsPerPixel == 16) {
      assert(lckScoped.stride >= size.w * 2);
      // stridewise copying
      for (std::uint32_t y = 0; y < size.h; ++y) {
        const std::uint8_t* ptrLine = ((const std::uint8_t*)lckScoped.ptrDataRoi) + y * lckScoped.stride;
        // uint16 is 2 bytes per pixel
        memcpy(dataPtr + (bytesPerRow * y), ptrLine, bytesPerRow);
      }
    } else if (volumeDims.bitsPerPixel == 8) {
      assert(lckScoped.stride >= size.w);
      // stridewise copying
      for (std::uint32_t y = 0; y < size.h; ++y) {
        const std::uint8_t* ptrLine = ((const std::uint8_t*)lckScoped.ptrDataRoi) + y * lckScoped.stride;
        uint16_t* destLine = reinterpret_cast<uint16_t*>(dataPtr + (bytesPerRow * y));
        for (size_t x = 0; x < size.w; ++x) {
          *destLine++ = *(ptrLine + x);
        }
      }
    }
    // else do nothing.
    // buffer is already initialized to zero,
    // and dimension validation earlier should prevent anything unintentional here.
  }
  return true;
}

uint32_t
FileReaderCzi::loadNumScenesCzi(const std::string& filepath)
{
  try {
    ScopedCziReader scopedReader(filepath);
    std::shared_ptr<libCZI::ICZIReader> cziReader = scopedReader.reader();

    auto statistics = cziReader->GetStatistics();
    int sceneStart = 0;
    int sceneSize = 0;
    bool scenesDefined = statistics.dimBounds.TryGetInterval(libCZI::DimensionIndex::S, &sceneStart, &sceneSize);
    if (!scenesDefined) {
      // assume just one?
      return 1;
    }
    return sceneSize;

  } catch (std::exception& e) {
    LOG_ERROR << e.what();
    LOG_ERROR << "Failed to read " << filepath;
    return 0;
  } catch (...) {
    LOG_ERROR << "Failed to read " << filepath;
    return 0;
  }
  return 0;
}

VolumeDimensions
FileReaderCzi::loadDimensionsCzi(const std::string& filepath, uint32_t scene)
{
  VolumeDimensions dims;
  try {
    ScopedCziReader scopedReader(filepath);
    std::shared_ptr<libCZI::ICZIReader> cziReader = scopedReader.reader();

    auto statistics = cziReader->GetStatistics();

    bool dims_ok = readCziDimensions(cziReader, filepath, statistics, dims, scene);
    if (!dims_ok) {
      return VolumeDimensions();
    }

    return dims;

  } catch (std::exception& e) {
    LOG_ERROR << e.what();
    LOG_ERROR << "Failed to read " << filepath;
    return VolumeDimensions();
  } catch (...) {
    LOG_ERROR << "Failed to read " << filepath;
    return VolumeDimensions();
  }
}

std::shared_ptr<ImageXYZC>
FileReaderCzi::loadCzi(const std::string& filepath, VolumeDimensions* outDims, uint32_t time, uint32_t scene)
{
  std::shared_ptr<ImageXYZC> emptyimage;

  auto tStart = std::chrono::high_resolution_clock::now();

  try {
    ScopedCziReader scopedReader(filepath);
    std::shared_ptr<libCZI::ICZIReader> cziReader = scopedReader.reader();

    auto statistics = cziReader->GetStatistics();

    VolumeDimensions dims;
    bool dims_ok = readCziDimensions(cziReader, filepath, statistics, dims, scene);
    if (!dims_ok) {
      return emptyimage;
    }
    int startT = 0, sizeT = 0;
    int startC = 0, sizeC = 0;
    int startZ = 0, sizeZ = 0;
    int startS = 0, sizeS = 0;
    bool hasT = statistics.dimBounds.TryGetInterval(libCZI::DimensionIndex::T, &startT, &sizeT);
    bool hasZ = statistics.dimBounds.TryGetInterval(libCZI::DimensionIndex::Z, &startZ, &sizeZ);
    bool hasC = statistics.dimBounds.TryGetInterval(libCZI::DimensionIndex::C, &startC, &sizeC);
    bool hasS = statistics.dimBounds.TryGetInterval(libCZI::DimensionIndex::S, &startS, &sizeS);

    if (!hasZ) {
      LOG_ERROR << "Agave can only read zstack volume data";
      return emptyimage;
    }
    if (dims.sizeC != sizeC || !hasC) {
      LOG_ERROR << "Inconsistent Channel count in czi file";
      return emptyimage;
    }

    size_t planesize = dims.sizeX * dims.sizeY * dims.bitsPerPixel / 8;
    uint8_t* data = new uint8_t[planesize * dims.sizeZ * dims.sizeC];
    memset(data, 0, planesize * dims.sizeZ * dims.sizeC);

    // stash it here in case of early exit, it will be deleted
    std::unique_ptr<uint8_t[]> smartPtr(data);

    uint8_t* destptr = data;

    // now ready to read channels one by one.
    libCZI::IntRect planeRect;
    if (hasS) {
      planeRect = statistics.sceneBoundingBoxes[startS + scene].boundingBoxLayer0;
    } else {
      planeRect = statistics.boundingBoxLayer0Only;
    }

    for (uint32_t channel = 0; channel < dims.sizeC; ++channel) {
      for (uint32_t slice = 0; slice < dims.sizeZ; ++slice) {
        destptr = data + planesize * (channel * dims.sizeZ + slice);

        // adjust coordinates by offsets from dims
        libCZI::CDimCoordinate planeCoord{ { libCZI::DimensionIndex::Z, (int)slice + startZ } };
        if (hasC) {
          planeCoord.Set(libCZI::DimensionIndex::C, (int)channel + startC);
        }
        if (hasT) {
          planeCoord.Set(libCZI::DimensionIndex::T, time + startT);
        }
        // since scene tiles can not overlap, passing the scene bounding box in to readCziPlane is enough produce the
        // scene, and I don't need to add Scene to the planeCoord.

        if (!readCziPlane(cziReader, planeRect, planeCoord, dims, destptr)) {
          return emptyimage;
        }
      }
    }

    auto tEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = tEnd - tStart;
    LOG_DEBUG << "CZI loaded in " << (elapsed.count() * 1000.0) << "ms";

    auto tStartImage = std::chrono::high_resolution_clock::now();

    // TODO: convert data to uint16_t pixels if not already.
    // we can release the smartPtr because ImageXYZC will now own the raw data memory
    ImageXYZC* im = new ImageXYZC(dims.sizeX,
                                  dims.sizeY,
                                  dims.sizeZ,
                                  dims.sizeC,
                                  IN_MEMORY_BPP, // dims.bitsPerPixel,
                                  smartPtr.release(),
                                  dims.physicalSizeX,
                                  dims.physicalSizeY,
                                  dims.physicalSizeZ);

    im->setChannelNames(dims.channelNames);

    tEnd = std::chrono::high_resolution_clock::now();
    elapsed = tEnd - tStartImage;
    LOG_DEBUG << "ImageXYZC prepared in " << (elapsed.count() * 1000.0) << "ms";

    elapsed = tEnd - tStart;
    LOG_DEBUG << "Loaded " << filepath << " in " << (elapsed.count() * 1000.0) << "ms";

    std::shared_ptr<ImageXYZC> sharedImage(im);
    if (outDims != nullptr) {
      *outDims = dims;
    }
    return sharedImage;

  } catch (std::exception& e) {
    LOG_ERROR << e.what();
    LOG_ERROR << "Failed to read " << filepath;
    return emptyimage;
  } catch (...) {
    LOG_ERROR << "Failed to read " << filepath;
    return emptyimage;
  }
}
