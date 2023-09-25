#include "FileReaderDICOM.h"

#include "ImageXYZC.h"
#include "Logging.h"

#include "dcmtk/dcmimgle/dcmimage.h"

#include <chrono>

// class DcmDicomDir is what you are looking for (see
// dcmdata/include/dcdicdir.h). This class allows you to read a DICOMDIR
// file and to browse through the logical structure (patient - study -
// series - instance) of the directory records.

// Use DcmDicomDir::getRootRecord() to get access to the root directory
// record, then use methods of class DcmDirectoryRecord to walk through
// the tree.

// Class DcmDirectoryRecord is derived from DcmItem, so you can treat it
// like any other dataset or sequence item. In addition, it offers
// the methods

// // provide directory record type
// virtual E_DirRecType getRecordType();

// // number of lower level directory records
// virtual unsigned long cardSub();

// // access lower level directory record
// virtual DcmDirectoryRecord* getSub(const unsigned long num);

// There are also a few methods for handling multiple-reference
// directory records (MRDR) but these should be rarely needed.

FileReaderDICOM::FileReaderDICOM(const std::string& filepath) {}
FileReaderDICOM::~FileReaderDICOM() {}

std::shared_ptr<ImageXYZC>
FileReaderDICOM::loadFromFile(const LoadSpec& loadSpec)
{
  std::string filepath = loadSpec.filepath;
  uint32_t scene = loadSpec.scene;
  uint32_t time = loadSpec.time;
  VolumeDimensions outDims;

  std::shared_ptr<ImageXYZC> emptyimage;

  auto tStart = std::chrono::high_resolution_clock::now();

  DicomImage* image = new DicomImage(loadSpec.filepath.c_str());
  if (image == nullptr) {
    LOG_ERROR << "Could not load DICOM image: " << loadSpec.filepath;
    return emptyimage;
  }
  VolumeDimensions dims;

  dims.sizeY = image->getHeight();
  dims.sizeX = image->getWidth();
  dims.sizeZ = image->getFrameCount();
  double aspectxy = image->getWidthHeightRatio();
  dims.physicalSizeX = aspectxy;
  dims.physicalSizeY = 1.0;
  dims.channelNames = { "DICOM" };

  uint32_t nch = loadSpec.channels.empty() ? dims.sizeC : loadSpec.channels.size();

  if (scene > 0) {
    LOG_WARNING << "Multiscene DICOM not supported. Using scene 0";
    scene = 0;
  }
  if (time > (int32_t)(dims.sizeT - 1)) {
    LOG_ERROR << "Time " << time << " exceeds time samples in file: " << dims.sizeT;
    return emptyimage;
  }

  size_t planesize_bytes = dims.sizeX * dims.sizeY * (ImageXYZC::IN_MEMORY_BPP / 8);
  size_t channelsize_bytes = planesize_bytes * dims.sizeZ;
  uint8_t* data = new uint8_t[channelsize_bytes * nch];
  memset(data, 0, channelsize_bytes * nch);
  // stash it here in case of early exit, it will be deleted
  std::unique_ptr<uint8_t[]> smartPtr(data);

  uint8_t* destptr = data;

  // still assuming 1 sample per pixel (scalar data) here.
  size_t rawPlanesize = dims.sizeX * dims.sizeY * (dims.bitsPerPixel / 8);
  // allocate temp data for one channel
  uint8_t* channelRawMem = new uint8_t[dims.sizeZ * rawPlanesize];
  memset(channelRawMem, 0, dims.sizeZ * rawPlanesize);

  // stash it here in case of early exit, it will be deleted
  std::unique_ptr<uint8_t[]> smartPtrTemp(channelRawMem);

  // now ready to read channels one by one.
  std::ifstream myFile(filepath, std::ios::in | std::ios::binary);
  for (uint32_t channel = 0; channel < nch; ++channel) {
    uint32_t channelToLoad = channel;
    if (!loadSpec.channels.empty()) {
      channelToLoad = loadSpec.channels[channel];
    }

    // read entire channel into its native size
    for (int k = 0; k < dims.sizeZ; k++) {

      uint16_t* pixelData = (uint16_t*)(image->getOutputData(16 /* bits */, k /* slice */));
      destptr = channelRawMem + k * rawPlanesize;
      memcpy(destptr, pixelData, rawPlanesize);
      delete[] pixelData;
    }

    // convert to our internal format (IN_MEMORY_BPP)
    // if (!convertChannelData(data + channel * channelsize_bytes, channelRawMem, dims)) {
    //   return emptyimage;
    // }
  }

  auto tEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = tEnd - tStart;
  LOG_DEBUG << "TIFF loaded in " << (elapsed.count() * 1000.0) << "ms";

  auto tStartImage = std::chrono::high_resolution_clock::now();

  // TODO: convert data to uint16_t pixels if not already.
  // we can release the smartPtr because ImageXYZC will now own the raw data memory
  ImageXYZC* im = new ImageXYZC(dims.sizeX,
                                dims.sizeY,
                                dims.sizeZ,
                                nch,
                                ImageXYZC::IN_MEMORY_BPP, // dims.bitsPerPixel,
                                smartPtr.release(),
                                dims.physicalSizeX,
                                dims.physicalSizeY,
                                dims.physicalSizeZ);

  std::vector<std::string> channelNames = dims.getChannelNames(loadSpec.channels);
  im->setChannelNames(channelNames);

  tEnd = std::chrono::high_resolution_clock::now();
  elapsed = tEnd - tStartImage;
  LOG_DEBUG << "ImageXYZC prepared in " << (elapsed.count() * 1000.0) << "ms";

  elapsed = tEnd - tStart;
  LOG_DEBUG << "Loaded " << filepath << " in " << (elapsed.count() * 1000.0) << "ms";

  std::shared_ptr<ImageXYZC> sharedImage(im);
  outDims = dims;

  return sharedImage;
}
VolumeDimensions
FileReaderDICOM::loadDimensions(const std::string& filepath, uint32_t scene)
{
  DicomImage* image = new DicomImage(filepath.c_str());
  VolumeDimensions dims;

  dims.sizeY = image->getHeight();
  dims.sizeX = image->getWidth();
  dims.sizeZ = image->getFrameCount();
  double aspectxy = image->getWidthHeightRatio();
  dims.physicalSizeX = aspectxy;
  dims.physicalSizeY = 1.0;
  dims.channelNames = { "DICOM" };

  dims.log();

  if (dims.validate()) {
    return dims;
  }
  return VolumeDimensions();
}

uint32_t
FileReaderDICOM::loadNumScenes(const std::string& filepath)
{
  return 1;
}
std::vector<MultiscaleDims>
FileReaderDICOM::loadMultiscaleDims(const std::string& filepath, uint32_t scene)
{
  VolumeDimensions dims = loadDimensions(filepath, scene);

  MultiscaleDims d;
  d.scale = { dims.physicalSizeZ, dims.physicalSizeY, dims.physicalSizeX };
  d.shape = { dims.sizeZ, dims.sizeY, dims.sizeX };
  d.dimensionOrder = { "Z", "Y", "X" };
  d.dtype = "uint16";
  d.path = filepath;
  d.channelNames = dims.channelNames;

  return { d };
}
