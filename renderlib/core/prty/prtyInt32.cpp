#include "core/prty/prtyInt32.hpp"

// #include "core/ch/chChunkParserUtil.hpp"
// #include "core/ch/chReader.hpp"
#include "serialize/docReader.h"
#include "serialize/docWriter.h"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyInt32::prtyInt32()
  : prtyPropertyTemplate("Int32", 0)
{
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyInt32::prtyInt32(const std::string& i_Name)
  : prtyPropertyTemplate(i_Name, 0)
{
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyInt32::prtyInt32(const std::string& i_Name, const int& i_InitialValue)
  : prtyPropertyTemplate(i_Name, i_InitialValue)
{
}

//--------------------------------------------------------------------
//	The type of property it is
//--------------------------------------------------------------------
const char*
prtyInt32::GetType()
{
  return "Int32";
}

//--------------------------------------------------------------------
//	operators
//--------------------------------------------------------------------
prtyInt32&
prtyInt32::operator=(const prtyInt32& i_Property)
{
  // copy base data
  prtyProperty::operator=(i_Property);

  SetValue(i_Property.GetValue());
  return *this;
}
prtyInt32&
prtyInt32::operator=(const int i_Value)
{
  SetValue(i_Value);
  return *this;
}

//--------------------------------------------------------------------
//	comparison operators
//--------------------------------------------------------------------
bool
prtyInt32::operator==(const prtyInt32& i_Property) const
{
  return (m_Value == i_Property.GetValue());
}
bool
prtyInt32::operator!=(const prtyInt32& i_Property) const
{
  return (m_Value != i_Property.GetValue());
}
bool
prtyInt32::operator==(const int i_Value) const
{
  return (m_Value == i_Value);
}
bool
prtyInt32::operator!=(const int i_Value) const
{
  return (m_Value != i_Value);
}

//--------------------------------------------------------------------
//	comparison operators
//--------------------------------------------------------------------
bool
prtyInt32::operator>(const int i_Value) const
{
  return (m_Value > i_Value);
}
bool
prtyInt32::operator>=(const int i_Value) const
{
  return (m_Value >= i_Value);
}
bool
prtyInt32::operator<(const int i_Value) const
{
  return (m_Value < i_Value);
}
bool
prtyInt32::operator<=(const int i_Value) const
{
  return (m_Value <= i_Value);
}
bool
prtyInt32::operator>(const prtyInt32& i_Value) const
{
  return (m_Value > i_Value.GetValue());
}
bool
prtyInt32::operator>=(const prtyInt32& i_Value) const
{
  return (m_Value >= i_Value.GetValue());
}
bool
prtyInt32::operator<(const prtyInt32& i_Value) const
{
  return (m_Value < i_Value.GetValue());
}
bool
prtyInt32::operator<=(const prtyInt32& i_Value) const
{
  return (m_Value <= i_Value.GetValue());
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyInt32::Read(docReader& io_Reader)
{
  int32_t temp;
  temp = io_Reader.readInt32();
  SetValue(temp);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyInt32::Write(docWriter& io_Writer) const
{
  io_Writer.writeInt32(GetPropertyName(), GetValue());
}
