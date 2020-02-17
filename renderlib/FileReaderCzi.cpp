#include "FileReaderCzi.h"

#include "BoundingBox.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "VolumeDimensions.h"

#include <libCZI/Src/libCZI/libCZI.h>

#include <QDomDocument>
#include <QElapsedTimer>

#include <boost/filesystem.hpp>

#include <map>
#include <set>

FileReaderCzi::FileReaderCzi() {}

FileReaderCzi::~FileReaderCzi() {}

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

float
requireFloatValue(QDomElement& el, float defaultVal)
{
  QString val = el.text();
  bool ok;
  float retval = val.toFloat(&ok);
  if (!ok) {
    retval = defaultVal;
  }
  return retval;
}

QDomElement
firstChild(QDomElement el, QString tag)
{
  QDomElement child = el.elementsByTagName(tag).at(0).toElement();
  if (child.isNull()) {
    LOG_ERROR << "No " << tag.toStdString() << "element in xml";
  }
  return child;
}

bool
readCziDimensions(const std::shared_ptr<libCZI::ICZIReader>& reader,
                  const std::string filepath,
                  libCZI::SubBlockStatistics& statistics,
                  VolumeDimensions& dims)
{
  // check for mosaic.  we can't (won't) handle those right now.
  if (statistics.maxMindex > 0) {
    LOG_ERROR << "CZI file is mosaic; mosaic reading not yet implemented";
    return false;
  }

  // metadata xml
  auto mds = reader->ReadMetadataSegment();
  std::shared_ptr<libCZI::ICziMetadata> md = mds->CreateMetaFromMetadataSegment();
  std::shared_ptr<libCZI::ICziMultiDimensionDocumentInfo> docinfo = md->GetDocumentInfo();

  docinfo->EnumDimensions([&](libCZI::DimensionIndex dimensionIndex) -> bool {
    std::shared_ptr<libCZI::IDimensionInfo> dimInfo = docinfo->GetDimensionInfo(dimensionIndex);
    return true;
  });

  libCZI::ScalingInfo scalingInfo = docinfo->GetScalingInfo();
  // convert meters to microns?
  dims.physicalSizeX = scalingInfo.scaleX * 1000000.0f;
  dims.physicalSizeY = scalingInfo.scaleY * 1000000.0f;
  dims.physicalSizeZ = scalingInfo.scaleZ * 1000000.0f;

  // get all dimension bounds and enumerate
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
    }
    return true;
  });

  libCZI::IntRect planebox = getSceneYXSize(statistics);
  dims.sizeX = planebox.w;
  dims.sizeY = planebox.h;

  std::string xml = md->GetXml();
  // convert to QString for convenience functions
  QString qxmlstr = QString::fromStdString(xml);
  QDomDocument czixml;
  bool ok = czixml.setContent(qxmlstr);
  if (!ok) {
    LOG_ERROR << "Bad CZI xml metadata content";
    return false;
  }

  QDomElement metadataEl = czixml.elementsByTagName("Metadata").at(0).toElement();
  if (metadataEl.isNull()) {
    LOG_ERROR << "No Metadata element in czi xml";
    return false;
  }
  QDomElement informationEl = firstChild(metadataEl, "Information");
  if (informationEl.isNull()) {
    return false;
  }
  QDomElement imageEl = firstChild(informationEl, "Image");
  if (imageEl.isNull()) {
    return false;
  }
  QDomElement dimensionsEl = firstChild(imageEl, "Dimensions");
  if (dimensionsEl.isNull()) {
    return false;
  }
  QDomElement channelsEl = firstChild(dimensionsEl, "Channels");
  if (channelsEl.isNull()) {
    return false;
  }
  QDomNodeList channelEls = channelsEl.elementsByTagName("Channel");
  std::vector<std::string> channelNames;
  for (int i = 0; i < channelEls.count(); ++i) {
    QDomNode node = channelEls.at(i);
    if (node.isElement()) {
      QDomElement el = node.toElement();
      channelNames.push_back(el.attribute("Name").toStdString());
    }
  }

  dims.channelNames = channelNames;

  libCZI::SubBlockInfo info;
  ok = reader->TryGetSubBlockInfoOfArbitrarySubBlockInChannel(0, info);
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
  reader->EnumSubset(&planeCoord, &planeRect, true, [&](int idx, const libCZI::SubBlockInfo& info) -> bool {
    // accept first subblock
    std::shared_ptr<libCZI::ISubBlock> subblock = reader->ReadSubBlock(idx);

    std::shared_ptr<libCZI::IBitmapData> bitmap = subblock->CreateBitmap();
    // and copy memory
    libCZI::IntSize size = bitmap->GetSize();
    {
      libCZI::ScopedBitmapLockerSP lckScoped{ bitmap };
      assert(lckScoped.stride >= size.w * 2);
      assert(lckScoped.ptrDataRoi == lckScoped.ptrData);
      assert(volumeDims.sizeX == size.w);
      assert(volumeDims.sizeY == size.h);
      size_t bytesPerRow = size.w * 2; // destination stride
      // stridewise copying
      for (std::uint32_t y = 0; y < size.h; ++y) {
        const std::uint8_t* ptrLine = ((const std::uint8_t*)lckScoped.ptrDataRoi) + y * lckScoped.stride;
        // uint16 is 2 bytes per pixel
        memcpy(dataPtr + (bytesPerRow * y), ptrLine, bytesPerRow);
      }
    }

    // stop iterating, on the assumption that there is only one subblock that fits this planecoordinate
    return false;
  });

  return true;
}

std::shared_ptr<ImageXYZC>
FileReaderCzi::loadCzi_4D(const std::string& filepath)
{
  std::shared_ptr<ImageXYZC> emptyimage;

  QElapsedTimer twhole;
  twhole.start();

  QElapsedTimer timer;
  timer.start();

  try {
    // get path as a wchar_t pointer for libCZI
    boost::filesystem::path fpath(filepath);
    const std::wstring widestr = fpath.wstring();

    std::shared_ptr<libCZI::IStream> stream = libCZI::CreateStreamFromFile(widestr.c_str());
    std::shared_ptr<libCZI::ICZIReader> cziReader = libCZI::CreateCZIReader();

    cziReader->Open(stream);

    auto statistics = cziReader->GetStatistics();

    VolumeDimensions dims;
    bool dims_ok = readCziDimensions(cziReader, filepath, statistics, dims);
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

    for (uint32_t channel = 0; channel < dims.sizeC; ++channel) {
      for (uint32_t slice = 0; slice < dims.sizeZ; ++slice) {
        destptr = data + planesize * (channel * dims.sizeZ + slice);

        // adjust coordinates by offsets from dims
        libCZI::CDimCoordinate planeCoord{ { libCZI::DimensionIndex::Z, (int)slice + startZ } };
        if (hasC) {
          planeCoord.Set(libCZI::DimensionIndex::C, (int)channel + startC);
        }
        if (hasS) {
          planeCoord.Set(libCZI::DimensionIndex::S, 0 + startS);
        }
        if (hasT) {
          planeCoord.Set(libCZI::DimensionIndex::T, 0 + startT);
        }

        if (!readCziPlane(cziReader, statistics.boundingBoxLayer0Only, planeCoord, dims, destptr)) {
          return emptyimage;
        }
      }
    }

    cziReader->Close();

    LOG_DEBUG << "CZI loaded in " << timer.elapsed() << "ms";

    // TODO: convert data to uint16_t pixels if not already.

    timer.start();
    // we can release the smartPtr because ImageXYZC will now own the raw data memory
    ImageXYZC* im = new ImageXYZC(dims.sizeX,
                                  dims.sizeY,
                                  dims.sizeZ,
                                  dims.sizeC,
                                  dims.bitsPerPixel,
                                  smartPtr.release(),
                                  dims.physicalSizeX,
                                  dims.physicalSizeY,
                                  dims.physicalSizeZ);
    LOG_DEBUG << "ImageXYZC prepared in " << timer.elapsed() << "ms";

    im->setChannelNames(dims.channelNames);

    LOG_DEBUG << "Loaded " << filepath << " in " << twhole.elapsed() << "ms";

    std::shared_ptr<ImageXYZC> sharedImage(im);
    return sharedImage;

  } catch (...) {
    LOG_ERROR << "Failed to read " << filepath;
    return emptyimage;
  }
}
