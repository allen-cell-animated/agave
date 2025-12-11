#include "core/prty/prtyInt8.hpp"

// #include "core/ch/chReader.hpp"
#include "serialize/docReader.h"
#include "serialize/docWriter.h"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyInt8::prtyInt8()
  : prtyPropertyTemplate("Int8", 0)
{
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyInt8::prtyInt8(const std::string& i_Name)
  : prtyPropertyTemplate(i_Name, 0)
{
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyInt8::prtyInt8(const std::string& i_Name, int8_t i_InitialValue)
  : prtyPropertyTemplate(i_Name, i_InitialValue)
{
}

//--------------------------------------------------------------------
//	The type of property it is
//--------------------------------------------------------------------
const char*
prtyInt8::GetType()
{
  return "Int8";
}

//--------------------------------------------------------------------
//	operators
//--------------------------------------------------------------------
prtyInt8&
prtyInt8::operator=(const prtyInt8& i_Property)
{
  // copy base data
  prtyProperty::operator=(i_Property);

  SetValue(i_Property.GetValue());
  return *this;
}
prtyInt8&
prtyInt8::operator=(const int8_t i_Value)
{
  SetValue(i_Value);
  return *this;
}

//--------------------------------------------------------------------
//	comparison operators
//--------------------------------------------------------------------
bool
prtyInt8::operator==(const prtyInt8& i_Property) const
{
  return (m_Value == i_Property.GetValue());
}
bool
prtyInt8::operator!=(const prtyInt8& i_Property) const
{
  return (m_Value != i_Property.GetValue());
}
bool
prtyInt8::operator==(const int8_t i_Value) const
{
  return (m_Value == i_Value);
}
bool
prtyInt8::operator!=(const int8_t i_Value) const
{
  return (m_Value != i_Value);
}

//--------------------------------------------------------------------
//	comparison operators
//--------------------------------------------------------------------
bool
prtyInt8::operator>(const int8_t i_Value) const
{
  return (m_Value > i_Value);
}
bool
prtyInt8::operator>=(const int8_t i_Value) const
{
  return (m_Value >= i_Value);
}
bool
prtyInt8::operator<(const int8_t i_Value) const
{
  return (m_Value < i_Value);
}
bool
prtyInt8::operator<=(const int8_t i_Value) const
{
  return (m_Value <= i_Value);
}
bool
prtyInt8::operator>(const prtyInt8& i_Value) const
{
  return (m_Value > i_Value.GetValue());
}
bool
prtyInt8::operator>=(const prtyInt8& i_Value) const
{
  return (m_Value >= i_Value.GetValue());
}
bool
prtyInt8::operator<(const prtyInt8& i_Value) const
{
  return (m_Value < i_Value.GetValue());
}
bool
prtyInt8::operator<=(const prtyInt8& i_Value) const
{
  return (m_Value <= i_Value.GetValue());
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyInt8::Read(docReader& io_Reader)
{
  int8_t temp;
  temp = io_Reader.readInt8();
  SetValue(temp);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyInt8::Write(docWriter& io_Writer) const
{
  io_Writer.writeInt8(GetPropertyName(), GetValue());
}
