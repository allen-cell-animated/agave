#pragma once

#include "IFileReader.h"
#include "VolumeDimensions.h"

#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

// must include after tensorstore so that tensorstore picks up its own internal json impl
#include "json/json.hpp"

#include <memory>
#include <string>

class ImageXYZC;

class FileReaderZarr : public IFileReader
{
public:
  FileReaderZarr(const std::string& filepath);
  virtual ~FileReaderZarr();

  bool supportChunkedLoading() const { return true; }

  std::shared_ptr<ImageXYZC> loadFromFile(const LoadSpec& loadSpec);
  VolumeDimensions loadDimensions(const std::string& filepath, uint32_t scene = 0);
  uint32_t loadNumScenes(const std::string& filepath);
  std::vector<MultiscaleDims> loadMultiscaleDims(const std::string& filepath, uint32_t scene = 0);

private:
  nlohmann::json jsonRead(const std::string& filepath);
  std::vector<std::string> getChannelNames(const std::string& filepath);

  nlohmann::json getMultiscales(nlohmann::json attrs);
  nlohmann::json getOmero(nlohmann::json attrs);
  std::string tensorstoreZarrDriverName();

  int m_zarrVersion;
  nlohmann::json m_zattrs;
  tensorstore::TensorStore<> m_store;
};
