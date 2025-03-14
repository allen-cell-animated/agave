#include "StringUtil.h"

#include "Logging.h"

#include <regex>
#include <sstream>

std::string
trim(const std::string& str, const std::string& whitespace)
{
  const auto strBegin = str.find_first_not_of(whitespace);
  if (strBegin == std::string::npos)
    return ""; // no content

  const auto strEnd = str.find_last_not_of(whitespace);
  const auto strRange = strEnd - strBegin + 1;

  return str.substr(strBegin, strRange);
}

bool
startsWith(const std::string mainStr, const std::string toMatch)
{
  // std::string::find returns 0 if toMatch is found at starting
  if (mainStr.find(toMatch) == 0)
    return true;
  else
    return false;
}

bool
endsWith(std::string const& value, std::string const& ending)
{
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void
split(const std::string& s, char delim, std::vector<std::string>& elems)
{
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

// multi lines split by newline
// each line split by =
std::map<std::string, std::string>
splitToNameValuePairs(const std::string& s)
{
  std::vector<std::string> sl;
  split(s, '\n', sl);

  // split each string into name/value pairs,
  // then look up as a map.
  std::map<std::string, std::string> pairs;
  for (int i = 0; i < sl.size(); ++i) {
    std::vector<std::string> namevalue;
    split(sl[i], '=', namevalue);
    if (namevalue.size() == 2) {
      pairs[namevalue[0]] = namevalue[1];
    } else if (namevalue.size() == 1) {
      pairs[namevalue[0]] = "";
    } else if (sl[i] == "") {
      // ignore empty line
    } else {
      // on error return empty map.
      LOG_ERROR << "Unexpected name/value pair: " << sl[i];
      return std::map<std::string, std::string>();
    }
  }
  return pairs;
}

std::string
escapePath(const std::string& path)
{
  return std::regex_replace(path, std::regex("\\\\"), "\\\\");
}