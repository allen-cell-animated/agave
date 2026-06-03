#include "CacheSettings.h"

#include "renderlib/CacheManager.h"
#include "renderlib/Logging.h"
#include "renderlib/SystemInfo.h"

#include <QDir>
#include <QFile>
#include <QStandardPaths>

#include <nlohmann/json.hpp>

namespace {

::CacheConfig
toRenderlibConfig(const CacheSettings& settings, const CacheSettingsData& data)
{
  CacheSettingsData normalized = data;
  normalized.cacheDir = settings.defaultSettings().cacheDir;

  ::CacheConfig config;
  config.enabled = normalized.enabled;
  config.enableDisk = normalized.enableDisk;
  config.cacheDir = normalized.cacheDir;

  std::uint64_t availableMem = SystemInfo::availableMemoryBytes();
  if (availableMem > 0) {
    config.maxRamBytes = std::min(normalized.maxRamBytes, availableMem);
    if (config.maxRamBytes < normalized.maxRamBytes) {
      LOG_WARNING << "Cache RAM limit reduced from requested " << normalized.maxRamBytes << " to " << config.maxRamBytes
                  << " bytes to fit available memory.";
    }
  } else {
    config.maxRamBytes = normalized.maxRamBytes;
  }

  std::uint64_t availableDisk = SystemInfo::availableDiskBytes(normalized.cacheDir);
  if (availableDisk > 0) {
    config.maxDiskBytes = std::min(normalized.maxDiskBytes, availableDisk);
    if (config.maxDiskBytes < normalized.maxDiskBytes && normalized.enableDisk) {
      LOG_WARNING << "Cache disk limit reduced from requested " << normalized.maxDiskBytes << " to "
                  << config.maxDiskBytes << " bytes to fit available disk space.";
    }
  } else {
    config.maxDiskBytes = normalized.maxDiskBytes;
  }

  if (!config.enableDisk) {
    config.maxDiskBytes = 0;
  }

  return config;
}

} // namespace

CacheSettings::CacheSettings() {}

CacheSettingsData
CacheSettings::defaultSettings() const
{
  CacheSettingsData data;
  QString baseDir = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
  if (baseDir.isEmpty()) {
    baseDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
  }
  if (baseDir.isEmpty()) {
    baseDir = QDir::currentPath();
  }
  data.cacheDir = QDir(baseDir).filePath("agave-cache").toStdString();
  return data;
}

std::string
CacheSettings::configPath() const
{
  QString baseDir = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
  if (baseDir.isEmpty()) {
    baseDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
  }
  if (baseDir.isEmpty()) {
    baseDir = QDir::currentPath();
  }
  QDir().mkpath(baseDir);
  return QDir(baseDir).filePath("cache_settings.json").toStdString();
}

CacheSettingsData
CacheSettings::load()
{
  CacheSettingsData data = defaultSettings();
  QString path = QString::fromStdString(configPath());
  QFile file(path);
  if (!file.exists()) {
    return data;
  }
  if (!file.open(QIODevice::ReadOnly)) {
    return data;
  }

  QByteArray raw = file.readAll();
  try {
    nlohmann::json doc = nlohmann::json::parse(raw.toStdString());
    if (doc.contains("enabled")) {
      data.enabled = doc["enabled"].get<bool>();
    }
    if (doc.contains("enableDisk")) {
      data.enableDisk = doc["enableDisk"].get<bool>();
    }
    if (doc.contains("maxRamBytes")) {
      data.maxRamBytes = doc["maxRamBytes"].get<std::uint64_t>();
    }
    if (doc.contains("maxDiskBytes")) {
      data.maxDiskBytes = doc["maxDiskBytes"].get<std::uint64_t>();
    }
  } catch (...) {
    return defaultSettings();
  }

  // Cache files are not user data. Always use AGAVE's platform-selected cache
  // directory and ignore custom paths from older settings files.
  data.cacheDir = defaultSettings().cacheDir;

  return data;
}

bool
CacheSettings::save(const CacheSettingsData& data) const
{
  nlohmann::json doc;
  doc["enabled"] = data.enabled;
  doc["enableDisk"] = data.enableDisk;
  doc["maxRamBytes"] = data.maxRamBytes;
  doc["maxDiskBytes"] = data.maxDiskBytes;

  QString path = QString::fromStdString(configPath());
  QFile file(path);
  if (!file.open(QIODevice::WriteOnly)) {
    return false;
  }
  std::string out = doc.dump(2);
  file.write(out.c_str(), static_cast<int>(out.size()));
  return true;
}

void
CacheSettings::applyToRenderlib(const CacheSettingsData& data) const
{
  ::CacheConfig config = toRenderlibConfig(*this, data);
  if (config.enableDisk && !config.cacheDir.empty()) {
    if (!CacheManager::canWriteCacheDir(config.cacheDir)) {
      LOG_WARNING << "Cache disk disabled: cache directory not writable: " << config.cacheDir;
      config.enableDisk = false;
      config.maxDiskBytes = 0;
    }
  }
  LOG_INFO << "Cache config: enabled=" << (config.enabled ? 1 : 0) << " ram_bytes=" << config.maxRamBytes
           << " disk_enabled=" << (config.enableDisk ? 1 : 0) << " disk_bytes=" << config.maxDiskBytes
           << " cache_dir=" << config.cacheDir;
  CacheManager::instance().setConfig(config);
  auto applied = CacheManager::instance().getConfig();
  LOG_INFO << "Cache config applied: enabled=" << (applied.enabled ? 1 : 0) << " ram_bytes=" << applied.maxRamBytes
           << " disk_enabled=" << (applied.enableDisk ? 1 : 0) << " disk_bytes=" << applied.maxDiskBytes
           << " cache_dir=" << applied.cacheDir;
}
