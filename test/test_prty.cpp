// create catch2 unit tests for core/prty/prtyProperty.h

#include <catch2/catch_test_macros.hpp>

#include "core/prty/prtyProperty.hpp"
#include "core/prty/prtyIntegerTemplate.hpp"
#include "core/prty/prtyText.hpp"
#include "core/prty/prtyPropertyTemplate.hpp"

#include <string>
#include <iostream>

TEST_CASE("prtyProperty", "[prtyProperty]")
{
  prtyInt8 p("int");

  // no init value
  // REQUIRE(p.get() == 0);
  p.SetValue(42);
  REQUIRE(p.GetValue() == 42);
  REQUIRE(p == 42);
  REQUIRE(p.GetPropertyName() == "int");
  REQUIRE(p != 43);

  prtyText p2("string", "hello");
  REQUIRE(p2.GetValue() == "hello");
  REQUIRE(p2.GetPropertyName() == "string");
  REQUIRE(p2 == "hello");
  REQUIRE(p2 != "world");
  p2.SetValue("world");
  REQUIRE(p2.GetValue() == "world");
}

// test prtyProperty callbacks

TEST_CASE("prtyProperty callbacks", "[prtyProperty]")
{

  prtyInt8 p("int");
  int count = 0;

  p.AddCallback(new prtyCallbackLambda([&count](prtyProperty*, bool) { count++; }));
  REQUIRE(count == 0);
  p.SetValue(42);
  REQUIRE(count == 1);
  p.SetValue(42);
  REQUIRE(count == 1);
  p.SetValue(43);
  REQUIRE(count == 2);
}

// test behavior of copying a prtyProperty with callbacks

TEST_CASE("prtyProperty copy", "[prtyProperty]")
{
  prtyInt8 p("int");
  int count = 0;
  p.AddCallback(new prtyCallbackLambda([&count](prtyProperty*, bool) { count++; }));
  REQUIRE(count == 0);
  p.SetValue(42);
  REQUIRE(count == 1);

  prtyInt8 p2 = p;
  REQUIRE(p2.GetValue() == p.GetValue());
  REQUIRE(p2.GetPropertyName() == p.GetPropertyName());
  // callback not copied
  REQUIRE(count == 1);
  p2.SetValue(43);
  REQUIRE(count == 1);
  p.SetValue(44);
  REQUIRE(count == 2);

  REQUIRE(p2.GetValue() == 43);
  REQUIRE(p.GetValue() == 44);
}

// test a prtyProperty with a struct type

struct Foo
{
  int x;
  int y;

  friend std::ostream& operator<<(std::ostream& os, const Foo& f)
  {
    os << "{ " << f.x << ", " << f.y << " }";
    return os;
  }

  friend bool operator==(const Foo& lhs, const Foo& rhs) { return lhs.x == rhs.x && lhs.y == rhs.y; }
  friend bool operator!=(const Foo& lhs, const Foo& rhs) { return !(lhs == rhs); }
};

TEST_CASE("prtyProperty struct", "[prtyProperty]")
{
  class prtyFoo : public prtyPropertyTemplate<Foo, const Foo&>
  {
  public:
    prtyFoo(const std::string& i_Name)
      : prtyPropertyTemplate<Foo, const Foo&>(i_Name, { 0, 0 })
    {
    }

    virtual const char* GetType() override { return "Foo"; }
    virtual void Read(docReader& io_Reader) override
    {
      // Implement reading from a reader if needed
    }
    virtual void Write(docWriter& io_Writer) const override
    {
      // Implement writing to a writer if needed
    }
  };
  prtyFoo p("foo");
  Foo f = { 1, 2 };
  p.SetValue(f);
  REQUIRE(p.GetValue().x == 1);
  REQUIRE(p.GetValue().y == 2);
  Foo f2 = { 3, 4 };
  p.SetValue(f2);
  REQUIRE(p.GetValue().x == 3);
  REQUIRE(p.GetValue().y == 4);
}
