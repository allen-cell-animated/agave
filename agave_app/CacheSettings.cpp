#include "CacheSettings.h"

#include "renderlib/CacheManager.h"
#include "renderlib/Logging.h"

#include <QDir>
#include <QFile>
#include <QStandardPaths>
#include <QStorageInfo>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

namespace {

::CacheConfig
toRenderlibConfig(const CacheSettings& settings, const CacheSettingsData& data)
{
  CacheSettingsData normalized = data;
  if (normalized.cacheDir.empty()) {
    normalized.cacheDir = settings.defaultSettings().cacheDir;
  }

  ::CacheConfig config;
  config.enabled = normalized.enabled;
  config.enableDisk = normalized.enableDisk;
  config.cacheDir = normalized.cacheDir;

  std::uint64_t availableMem = settings.availableMemoryBytes();
  if (availableMem > 0) {
    config.maxRamBytes = std::min(normalized.maxRamBytes, availableMem);
  } else {
    config.maxRamBytes = normalized.maxRamBytes;
  }

  std::uint64_t availableDisk = settings.availableDiskBytes(normalized.cacheDir);
  if (availableDisk > 0) {
    config.maxDiskBytes = std::min(normalized.maxDiskBytes, availableDisk);
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
    if (doc.contains("cacheDir")) {
      data.cacheDir = doc["cacheDir"].get<std::string>();
    }
  } catch (...) {
    return defaultSettings();
  }

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
  doc["cacheDir"] = data.cacheDir;

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
    if (!canWriteCacheDir(config.cacheDir)) {
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

std::uint64_t
CacheSettings::availableMemoryBytes() const
{
#ifdef _WIN32
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  if (GlobalMemoryStatusEx(&statex)) {
    return static_cast<std::uint64_t>(statex.ullAvailPhys);
  }
  return 0;
#elif defined(__linux__)
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    return static_cast<std::uint64_t>(info.freeram) * static_cast<std::uint64_t>(info.mem_unit);
  }
  return 0;
#else
  return 0;
#endif
}

std::uint64_t
CacheSettings::availableDiskBytes(const std::string& path) const
{
  if (path.empty()) {
    return 0;
  }
  QStorageInfo storage(QString::fromStdString(path));
  if (!storage.isValid() || !storage.isReady()) {
    return 0;
  }
  return static_cast<std::uint64_t>(storage.bytesAvailable());
}

bool
CacheSettings::canWriteCacheDir(const std::string& path) const
{
  if (path.empty()) {
    return false;
  }

  QString qPath = QString::fromStdString(path);
  QDir dir(qPath);
  if (!dir.exists()) {
    if (!dir.mkpath(".")) {
      return false;
    }
  }

  QString testPath = dir.filePath(".agave_cache_write_test");
  QFile testFile(testPath);
  if (!testFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
    return false;
  }
  testFile.write("test");
  testFile.close();
  return testFile.remove();
}
