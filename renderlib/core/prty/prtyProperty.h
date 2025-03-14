#pragma once

#include "Logging.h"

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <type_traits>

// only enable this to verbosely dump every single property setter.
// This implies that the value type must be streamable to std::ostream.
#ifndef PRTY_PROPERTY_DEBUG
#define PRTY_PROPERTY_DEBUG 0
#endif

// a property has a name and a value.
// the init value is default-constructed or passed in.
// The value type must be copyable, assignable, and default-constructible.
// The value must also be streamable to std::ostream.
template<typename T,
         typename = std::enable_if_t<
           std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T> && std::is_default_constructible_v<T> &&
           std::is_move_constructible_v<T> && std::is_move_assignable_v<T> &&
           std::is_convertible_v<decltype(std::declval<std::ostream&>() << std::declval<T>()), std::ostream&>>>
class prtyProperty
{
public:
  prtyProperty(std::string name)
    : name(name)
  {
  }

  prtyProperty(std::string name, const T& val)
    : name(name)
    , value(val)
  {
  }

  virtual ~prtyProperty() {}

  void set(const T& val, bool isFromUI = false)
  {
    // TODO - check if the value has changed
    // TODO - do we notify if value hasn't changed?

#if PRTY_PROPERTY_DEBUG
    LOG_INFO << "Property " << name << " set from " << value << " to " << val;
#endif

    value = val;

    // call all callbacks
    notifyAll(isFromUI);
  }

  std::string getName() const { return name; }

  // copy????
  T get() const { return value; }

  // non-copy, just use paren operator
  T& operator()() const { return value; }

  // set up the rule of 5
  prtyProperty(const prtyProperty& other)
    : name(other.name)
    , value(other.value)
  {
  }

  prtyProperty& operator=(const prtyProperty& other)
  {
    if (this != &other) {
      name = other.name;
      value = other.value;
    }
    return *this;
  }

  prtyProperty(prtyProperty&& other)
    : name(std::move(other.name))
    , value(std::move(other.value))
  {
  }

  prtyProperty& operator=(prtyProperty&& other)
  {
    if (this != &other) {
      name = std::move(other.name);
      value = std::move(other.value);
    }
    return *this;
  }

  // direct assignment from a value
  prtyProperty& operator=(const T& val)
  {
    value = val;
    return *this;
  }

  // callback for when the property's value is set.
  // The boolean flag is true if the change is coming from the
  // user interface and therefore should mark the document
  // containing the property as dirty.
  typedef std::function<void(prtyProperty<T>*, bool)> prtyPropertyCallback;

  void addCallback(prtyPropertyCallback cb) { callbacks.push_back(cb); }
  void removeCallback(prtyPropertyCallback cb)
  {
    auto it = std::find(callbacks.begin(), callbacks.end(), cb);
    if (it != callbacks.end()) {
      callbacks.erase(it);
    }
  }

  void notifyAll(bool isFromUI = false)
  {
    for (auto& cb : callbacks) {
      cb(this, isFromUI);
    }
  }

private:
  std::string name;
  T value;

  std::vector<prtyPropertyCallback> callbacks;
};
