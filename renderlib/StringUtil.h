#pragma once

#include <string>
#include <vector>
#include <map>

// TODO: move into a String Utils
static std::string
trim(const std::string& str, const std::string& whitespace = " \t\r\n");

// TODO: move into a String Utils
static bool
startsWith(const std::string mainStr, const std::string toMatch);

// TODO: move into a String Utils
static bool
endsWith(std::string const& value, std::string const& ending);

// TODO: move into a String Utils
static void
split(const std::string& s, char delim, std::vector<std::string>& elems);

// multi lines split by newline
// each line split by =
std::map<std::string, std::string>
splitToNameValuePairs(const std::string& s);