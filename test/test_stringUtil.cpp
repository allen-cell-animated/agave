#include <catch2/catch_test_macros.hpp>

#include "renderlib/StringUtil.h"

TEST_CASE("StringUtil", "[StringUtil]")
{
  SECTION("trim")
  {
    REQUIRE(trim("  hello  ") == "hello");
    REQUIRE(trim("  hello") == "hello");
    REQUIRE(trim("hello  ") == "hello");
    REQUIRE(trim("hello") == "hello");
    REQUIRE(trim("  ") == "");
    REQUIRE(trim("") == "");
    REQUIRE(trim("  \t\n\r  ") == "");
    REQUIRE(trim("foobarfoobar", "foo") == "barfoobar");
    REQUIRE(trim("foobarfoo", "foo") == "bar");
    REQUIRE(trim("foobaroof", "foo") == "bar");
  }
  SECTION("startsWith")
  {
    REQUIRE(startsWith("hello", "he") == true);
    REQUIRE(startsWith("hello", "hello") == true);
    REQUIRE(startsWith("hello", "hello ") == false);
    REQUIRE(startsWith("hello", "hellox") == false);
    REQUIRE(startsWith("hello", "xhello") == false);
    REQUIRE(startsWith("hello", "x") == false);
    REQUIRE(startsWith("hello", "") == true);
    REQUIRE(startsWith("", "hello") == false);
    REQUIRE(startsWith("", "") == true);
  }
  SECTION("endsWith")
  {
    REQUIRE(endsWith("hello", "lo") == true);
    REQUIRE(endsWith("hello", "hello") == true);
    REQUIRE(endsWith("hello", " hello") == false);
    REQUIRE(endsWith("hello", "xhello") == false);
    REQUIRE(endsWith("hello", "x") == false);
    REQUIRE(endsWith("hello", "") == true);
    REQUIRE(endsWith("", "hello") == false);
    REQUIRE(endsWith("", "") == true);
  }
  SECTION("split")
  {
    std::vector<std::string> elems;
    split("hello world", ' ', elems);
    REQUIRE(elems.size() == 2);
    REQUIRE(elems[0] == "hello");
    REQUIRE(elems[1] == "world");

    elems.clear();
    split("hello world", 'x', elems);
    REQUIRE(elems.size() == 1);
    REQUIRE(elems[0] == "hello world");

    elems.clear();
    split("hello world", 'o', elems);
    REQUIRE(elems.size() == 3);
    REQUIRE(elems[0] == "hell");
    REQUIRE(elems[1] == " w");
    REQUIRE(elems[2] == "rld");

    elems.clear();
    split("name=value", '=', elems);
    REQUIRE(elems.size() == 2);
    REQUIRE(elems[0] == "name");
    REQUIRE(elems[1] == "value");

    elems.clear();
    split("name=value\nname2=value2\n", '\n', elems);
    REQUIRE(elems.size() == 2);
    REQUIRE(elems[0] == "name=value");
    REQUIRE(elems[1] == "name2=value2");
  }

  SECTION("splitToNameValuePairs")
  {
    std::map<std::string, std::string> pairs;

    pairs = splitToNameValuePairs("name");
    REQUIRE(pairs.size() == 1);
    REQUIRE(pairs["name"] == "");

    pairs = splitToNameValuePairs("name=");
    REQUIRE(pairs.size() == 1);
    REQUIRE(pairs["name"] == "");

    pairs = splitToNameValuePairs("name=value");
    REQUIRE(pairs.size() == 1);
    REQUIRE(pairs["name"] == "value");

    pairs = splitToNameValuePairs("name=value\nname2=value2\n");
    REQUIRE(pairs.size() == 2);
    REQUIRE(pairs["name"] == "value");
    REQUIRE(pairs["name2"] == "value2");

    pairs = splitToNameValuePairs("name=value\nname2=value2");
    REQUIRE(pairs.size() == 2);
    REQUIRE(pairs["name"] == "value");
    REQUIRE(pairs["name2"] == "value2");

    pairs = splitToNameValuePairs("name=value\nname2=value2\nname3=value3\n\n");
    REQUIRE(pairs.size() == 3);
    REQUIRE(pairs["name"] == "value");
    REQUIRE(pairs["name2"] == "value2");
    REQUIRE(pairs["name3"] == "value3");
  }
}
