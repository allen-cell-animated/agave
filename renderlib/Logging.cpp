#include "Logging.h"

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <filesystem>
#include <fstream>
#include <ostream>
#if defined(__APPLE__) || defined(__linux__)
#include <pwd.h>
#include <unistd.h>
#endif

// the logs are written to LOGFILE
#define LOGFILE "logfile.log"

static std::filesystem::path sLogFileDirectory = "";
static spdlog::logger* sLogger = nullptr;

void
Logging::Enable(bool enabled)
{
  spdlog::set_level(enabled ? spdlog::level::trace : spdlog::level::off);
}

std::filesystem::path
getLogPath()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  const char* rootdir = getenv("LOCALAPPDATA");
  return std::filesystem::path(rootdir) / "AllenInstitute" / "agave";
#elif __APPLE__
  const char* rootdir = getenv("HOME");
  if (!rootdir) {
    struct passwd* pwd = getpwuid(getuid());
    if (pwd)
      rootdir = pwd->pw_dir;
  }
  return std::filesystem::path(rootdir) / "Library" / "Logs" / "AllenInstitute" / "agave";
#elif __linux__
  const char* rootdir = getenv("HOME");
  if (!rootdir) {
    struct passwd* pwd = getpwuid(getuid());
    if (pwd)
      rootdir = pwd->pw_dir;
  }
  return std::filesystem::path(rootdir) / ".agave";
#else
#error "Unknown compiler"
#endif
}

void
Logging::Init()
{
  sLogFileDirectory = getLogPath();
  // make dir if doesn't exist.  throws on error
  std::filesystem::create_directories(sLogFileDirectory);
  std::filesystem::path logFilePath = sLogFileDirectory / LOGFILE;

  // 1. log to stdout
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

  // 2. log to file
  static const size_t MAX_LOG_FILE_SIZE_BYTES = 1024 * 1024 * 4; // 5 MB
  auto file_sink =
    std::make_shared<spdlog::sinks::rotating_file_sink_mt>(logFilePath.string(), MAX_LOG_FILE_SIZE_BYTES, 3);

  // unify the two loggers as the default single logger
  sLogger = new spdlog::logger("agave", { console_sink, file_sink });
  spdlog::set_default_logger(std::shared_ptr<spdlog::logger>(sLogger));
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
  spdlog::flush_on(spdlog::level::trace);

  Logging::Enable(true);

  LOG_INFO << "Logging Init DONE";
}
