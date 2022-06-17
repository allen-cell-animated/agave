#pragma once

#include <spdlog/spdlog.h>

#include <sstream>

#define SPDLOG_LOGGER_STREAM(log, lvl)                                                                                 \
  log && log->should_log(lvl) &&                                                                                       \
    LogStream(log, lvl, spdlog::source_loc{ __FILE__, __LINE__, SPDLOG_FUNCTION }) == LogLine()

class LogLine
{
  std::ostringstream m_ss;

public:
  LogLine() {}
  template<typename T>
  LogLine& operator<<(const T& t)
  {
    m_ss << t;
    return *this;
  }
  std::string str() const { return m_ss.str(); }
};

class LogStream
{
  spdlog::logger* m_log;
  spdlog::level::level_enum m_lvl;
  spdlog::source_loc m_loc;

public:
  LogStream(spdlog::logger* log, spdlog::level::level_enum lvl, spdlog::source_loc loc)
    : m_log{ log }
    , m_lvl{ lvl }
    , m_loc{ loc }
  {}
  bool operator==(const LogLine& line)
  {
    m_log->log(m_loc, m_lvl, "{}", line.str());
    return true;
  }
};

// specific log implementation macros

#define LOG(x) SPDLOG_LOGGER_STREAM(spdlog::default_logger_raw(), x)

#define LOG_TRACE LOG(spdlog::level::trace)
#define LOG_DEBUG LOG(spdlog::level::debug)
#define LOG_INFO LOG(spdlog::level::info)
#define LOG_WARNING LOG(spdlog::level::warn)
#define LOG_ERROR LOG(spdlog::level::err)
#define LOG_FATAL LOG(spdlog::level::critical)

namespace Logging {

// must be called early at app startup to ensure safety.
void
Init();

void
Enable(bool enabled);

};
