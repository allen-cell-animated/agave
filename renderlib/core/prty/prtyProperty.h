#pragma once

#include <string>
#include <type_traits>

// a property has a name and a value.
// the init value is default-constructed or passed in.
// The value type must be copyable, assignable, and default-constructible.
template <typename T, typename = std::enable_if_t<
std::is_copy_constructible_v<T> &&
std::is_copy_assignable_v<T> &&
std::is_default_constructible_v<T>>>
class prtyProperty {
public:
    prtyProperty(std::string name) : name(name) {}
    prtyProperty(std::string name, const T& val) : name(name), value(val) {}
    virtual ~prtyProperty() {}
  void set(const T& val) {value=val; }
  // copy????
  T get() const { return value; }
  // non-copy, just use paren operator
  T& operator() const { return value; }


  // set up the rule of 5
  prtyProperty(const prtyProperty& other) : name(other.name), value(other.value) {}
  prtyProperty& operator=(const prtyProperty& other) {
    if (this != &other) {
      name = other.name;
      value = other.value;
    }
    return *this;
  }
  prtyProperty(prtyProperty&& other) : name(std::move(other.name)), value(std::move(other.value)) {}
  prtyProperty& operator=(prtyProperty&& other) {
    if (this != &other) {
      name = std::move(other.name);
      value = std::move(other.value);
    }
    return *this;
  }

  // direct assignment from a value
  prtyProperty& operator=(const T& val) {
    value = val;
    return *this;
  }

private:
  std::string name;
  T value;
};
