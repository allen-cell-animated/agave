// create catch2 unit tests for core/prty/prtyProperty.h

#include <catch2/catch_test_macros.hpp>

#include "core/prty/prtyProperty.h"

#include <string>

TEST_CASE("prtyProperty", "[prtyProperty]")
{
  prtyProperty<int> p("int");

  // no init value
  // REQUIRE(p.get() == 0);
  p.set(42);
  REQUIRE(p.get() == 42);

  prtyProperty<std::string> p2("string", "hello");
  REQUIRE(p2.get() == "hello");
  p2.set("world");
  REQUIRE(p2.get() == "world");
}

// test prtyProperty callbacks

TEST_CASE("prtyProperty callbacks", "[prtyProperty]")
{
  prtyProperty<int> p("int");
  int count = 0;
  p.addCallback([&count](prtyProperty<int>*, bool) { count++; });
  REQUIRE(count == 0);
  p.set(42);
  REQUIRE(count == 1);
  // callback fires even if value doesn't change
  p.set(42);
  REQUIRE(count == 2);
  p.set(43);
  REQUIRE(count == 3);
}

// test behavior of copying a prtyProperty with callbacks

TEST_CASE("prtyProperty copy", "[prtyProperty]")
{
  prtyProperty<int> p("int");
  int count = 0;
  p.addCallback([&count](prtyProperty<int>*, bool) { count++; });
  REQUIRE(count == 0);
  p.set(42);
  REQUIRE(count == 1);

  prtyProperty<int> p2 = p;
  REQUIRE(p2.get() == p.get());
  REQUIRE(p2.getName() == p.getName());
  // callback not copied
  REQUIRE(count == 1);
  p2.set(43);
  REQUIRE(count == 1);
  p.set(44);
  REQUIRE(count == 2);

  REQUIRE(p2.get() == 43);
  REQUIRE(p.get() == 44);
}

// test a prtyProperty with a struct type

struct Foo
{
  int x;
  int y;
};

TEST_CASE("prtyProperty struct", "[prtyProperty]")
{
  prtyProperty<Foo> p("foo");
  Foo f = { 1, 2 };
  p.set(f);
  REQUIRE(p.get().x == 1);
  REQUIRE(p.get().y == 2);
  Foo f2 = { 3, 4 };
  p.set(f2);
  REQUIRE(p.get().x == 3);
  REQUIRE(p.get().y == 4);
}
