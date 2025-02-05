#pragma once

#include <string>
#include <vector>
#include <map>

std::string
trim(const std::string& str, const std::string& whitespace = " \t\r\n");

bool
startsWith(const std::string mainStr, const std::string toMatch);

bool
endsWith(std::string const& value, std::string const& ending);

void
split(const std::string& s, char delim, std::vector<std::string>& elems);

// multi lines split by newline
// each line split by =
std::map<std::string, std::string>
splitToNameValuePairs(const std::string& s);

std::string
escapePath(const std::string& path);
