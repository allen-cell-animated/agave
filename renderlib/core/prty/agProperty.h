#pragma once

#include "Logging.h"

#include "core/prty/agPropertyCallback.hpp"
#include "core/undo/undoUndoOperation.hpp"

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

class agBaseProperty
{
public:
  // Base class for properties, can be used to store common flags or data.
  // This class is not intended to be instantiated directly.
  agBaseProperty()
    : m_bVisible(true)
  {
    m_Flags.m_bAnimatable = false;
    m_Flags.m_bScripted = false;
    m_Flags.m_bValueVariation = false;
  }
  agBaseProperty(const std::string& i_Name)
    : name(i_Name)
    , m_bVisible(true)
  {
    m_Flags.m_bAnimatable = false;
    m_Flags.m_bScripted = false;
    m_Flags.m_bValueVariation = false;
  }

  virtual ~agBaseProperty() {}

  agBaseProperty& operator=(const agBaseProperty& i_Property);

  const std::string& GetPropertyName() const;
  void SetPropertyName(const std::string& i_Name);

  //--------------------------------------------------------------------
  // Animatable - can this property be animated?
  //--------------------------------------------------------------------
  bool IsAnimatable() const;
  void SetAnimatable(bool i_bVal);

  //--------------------------------------------------------------------
  // Scripted - Is this property currently being animated?
  //--------------------------------------------------------------------
  bool IsScripted() const;
  void SetScripted(bool i_bVal);

  //--------------------------------------------------------------------
  // ValueVariation - Does the current value vary from the scripted value?
  //--------------------------------------------------------------------
  bool HasValueVariation() const;
  void SetValueVariation(bool i_bVal);

  //--------------------------------------------------------------------
  // AddCallback using interface.
  //
  //	NOTE: This property will take ownership of the callback
  //	pointer and will delete it in its destructor! It should be
  //	created on the heap and probably should use the
  //	prtyCallbackWrapper template.
  //--------------------------------------------------------------------
  void AddCallback(agPropertyCallback* i_pCallback);

  //--------------------------------------------------------------------
  // AddCallback shared pointer variation for when you want to
  //	keep a copy of the callback also.
  //--------------------------------------------------------------------
  void AddCallback(std::shared_ptr<agPropertyCallback> i_pCallback);

  //--------------------------------------------------------------------
  // RemoveCallback by shared_ptr. You should be keeping a shared_ptr
  //	to the callback passed to AddCallback() if you want to remove
  //	it later. Don't use the direct pointer function when adding
  //	if you want to keep a copy.
  //--------------------------------------------------------------------
  void RemoveCallback(std::shared_ptr<agPropertyCallback> i_pCallback);

  //--------------------------------------------------------------------
  //	Call all the callbacks -
  //		i_bWithUndo should be set to (i_Undoable != eNoUndo)
  //	in derived classes SetValue() functions.
  //--------------------------------------------------------------------
  void NotifyCallbacksPropertyChanged(bool i_bWithUndo);

  //--------------------------------------------------------------------
  //	The type of property it is
  //--------------------------------------------------------------------
  virtual const char* GetType() = 0;

  //--------------------------------------------------------------------
  // Create an undo operation of the correct type for this
  // property. A reference to this property should be passed in.
  // Ownership passes to the caller.
  //--------------------------------------------------------------------
  //  virtual undoUndoOperation* CreateUndoOperation(std::shared_ptr<prtyPropertyReference> i_pPropertyRef) = 0;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  // virtual void Read(chReader& io_Reader) = 0;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  // virtual void Write(chWriter& io_Writer) const = 0;

  //--------------------------------------------------------------------
  //  In some cases we want the form builder to check whether or not
  // this property should be visible.
  //--------------------------------------------------------------------
  void SetVisible(bool i_bVisible);
  bool GetVisible() const;

protected:
  std::string name;
  std::vector<std::shared_ptr<agPropertyCallback>> m_CallbackObjects;
  bool m_bVisible; // is this property visible in the UI?
  // Flags for the property
  struct
  {
    bool m_bAnimatable : 1;     // can this property be animated?
    bool m_bScripted : 1;       // Is this property currently being animated?
    bool m_bValueVariation : 1; // Does the current value vary from the scripted value?
  } m_Flags;
};

// a property has a name and a value.
// the init value is default-constructed or passed in.
// The value type must be copyable, assignable, and default-constructible.
// The value must also be streamable to std::ostream.
template<typename T,
         typename = std::enable_if_t<
           std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T> && std::is_default_constructible_v<T> &&
           std::is_move_constructible_v<T> && std::is_move_assignable_v<T> &&
           std::is_convertible_v<decltype(std::declval<std::ostream&>() << std::declval<T>()), std::ostream&>>>
class agProperty : public agBaseProperty
{
public:
  agProperty(std::string name, const T& val)
    : agBaseProperty(name)
    , value(val)
  {
  }

  virtual ~agProperty() {}

  void set(const T& val, bool i_bDirty = false)
  {
    // TODO - check if the value has changed
    // TODO - do we notify if value hasn't changed?

#if PRTY_PROPERTY_DEBUG
    LOG_INFO << "Property " << name << " set from " << value << " to " << val;
#endif
    if (value != val) {
      value = val;

      // call all callbacks
      NotifyCallbacksPropertyChanged(i_bDirty);
    }
  }

  std::string getName() const { return name; }

  // copy????
  T get() const { return value; }

  // non-copy, just use paren operator
  T& operator()() const { return value; }

  // set up the rule of 5
  agProperty(const agProperty& other)
    : agBaseProperty(other.name)
    , value(other.value)
  {
  }

  agProperty& operator=(const agProperty& other)
  {
    if (this != &other) {
      name = other.name;
      set(other.value); // use set to notify callbacks
    }
    return *this;
  }

  // direct assignment from a value
  agProperty& operator=(const T& value)
  {
    set(value); // use set to notify callbacks
    return *this;
  }

  // TODO should we implement move semantics?
  //   agProperty(agProperty&& other)
  //     : name(std::move(other.name))
  //     , value(std::move(other.value))
  //   {
  //   }

  //   agProperty& operator=(agProperty&& other)
  //   {
  //     if (this != &other) {
  //       name = std::move(other.name);
  //       set(std::move(other.value)); // use set to notify callbacks
  //     }
  //     return *this;
  //   }

private:
  T value;
};
