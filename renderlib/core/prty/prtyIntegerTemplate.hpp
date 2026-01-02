#pragma once

#include "core/prty/prtyPropertyTemplate.hpp"
#include "serialize/docReader.h"
#include "serialize/docWriter.h"
#include <type_traits>

//============================================================================
// Template class for integer properties (signed and unsigned)
// This eliminates the need for separate prtyInt8, prtyInt16, prtyInt32, etc.
//============================================================================
template<typename T>
class prtyIntegerTemplate : public prtyPropertyTemplate<T, T>
{
  static_assert(std::is_integral_v<T>, "prtyIntegerTemplate only works with integral types");

public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyIntegerTemplate()
    : prtyPropertyTemplate<T, T>(GetTypeName(), T{})
  {
  }

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyIntegerTemplate(const std::string& i_Name)
    : prtyPropertyTemplate<T, T>(i_Name, T{})
  {
  }

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyIntegerTemplate(const std::string& i_Name, T i_InitialValue)
    : prtyPropertyTemplate<T, T>(i_Name, i_InitialValue)
  {
  }

  //--------------------------------------------------------------------
  //	The type of property it is
  //--------------------------------------------------------------------
  virtual const char* GetType() override { return GetTypeName(); }

  //--------------------------------------------------------------------
  //	operators
  //--------------------------------------------------------------------
  prtyIntegerTemplate& operator=(const prtyIntegerTemplate& i_Property)
  {
    // copy base data
    prtyProperty::operator=(i_Property);
    this->SetValue(i_Property.GetValue());
    return *this;
  }

  prtyIntegerTemplate& operator=(T i_Value)
  {
    this->SetValue(i_Value);
    return *this;
  }

  //--------------------------------------------------------------------
  //	comparison operators
  //--------------------------------------------------------------------
  bool operator==(const prtyIntegerTemplate& i_Property) const { return (this->m_Value == i_Property.GetValue()); }

  bool operator!=(const prtyIntegerTemplate& i_Property) const { return (this->m_Value != i_Property.GetValue()); }

  bool operator==(T i_Value) const { return (this->m_Value == i_Value); }

  bool operator!=(T i_Value) const { return (this->m_Value != i_Value); }

  //--------------------------------------------------------------------
  //	comparison operators
  //--------------------------------------------------------------------
  bool operator>(T i_Value) const { return (this->m_Value > i_Value); }

  bool operator>=(T i_Value) const { return (this->m_Value >= i_Value); }

  bool operator<(T i_Value) const { return (this->m_Value < i_Value); }

  bool operator<=(T i_Value) const { return (this->m_Value <= i_Value); }

  bool operator>(const prtyIntegerTemplate& i_Value) const { return (this->m_Value > i_Value.GetValue()); }

  bool operator>=(const prtyIntegerTemplate& i_Value) const { return (this->m_Value >= i_Value.GetValue()); }

  bool operator<(const prtyIntegerTemplate& i_Value) const { return (this->m_Value < i_Value.GetValue()); }

  bool operator<=(const prtyIntegerTemplate& i_Value) const { return (this->m_Value <= i_Value.GetValue()); }

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Read(docReader& io_Reader) override
  {
    T temp;
    if constexpr (std::is_signed_v<T>) {
      temp = io_Reader.readInt<T>(this->GetPropertyName());
    } else {
      temp = io_Reader.readUint<T>(this->GetPropertyName());
    }
    this->SetValue(temp);
  }

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Write(docWriter& io_Writer) const override
  {
    if constexpr (std::is_signed_v<T>) {
      io_Writer.writeInt<T>(this->GetPropertyName(), this->GetValue());
    } else {
      io_Writer.writeUint<T>(this->GetPropertyName(), this->GetValue());
    }
  }

private:
  static constexpr const char* GetTypeName()
  {
    if constexpr (std::is_same_v<T, int8_t>) {
      return "Int8";
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return "Int16";
    } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) {
      return "Int32";
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return "Int64";
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      return "Uint8";
    } else if constexpr (std::is_same_v<T, uint16_t>) {
      return "Uint16";
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      return "Uint32";
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return "Uint64";
    } else {
      return "Integer";
    }
  }
};

// Convenient type aliases for common integer types
using prtyInt8 = prtyIntegerTemplate<int8_t>;
using prtyInt16 = prtyIntegerTemplate<int16_t>;
using prtyInt32 = prtyIntegerTemplate<int32_t>;
using prtyInt64 = prtyIntegerTemplate<int64_t>;
using prtyUint8 = prtyIntegerTemplate<uint8_t>;
using prtyUint16 = prtyIntegerTemplate<uint16_t>;
using prtyUint32 = prtyIntegerTemplate<uint32_t>;
using prtyUint64 = prtyIntegerTemplate<uint64_t>;
