#include "core/prty/prtyFloat.hpp"
#include "core/prty/prtyUnits.hpp"

// #include "core/ch/chReader.hpp"
#include "serialize/docWriter.h"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyFloat::prtyFloat()
  : prtyPropertyTemplate("Float", 0.0f)
  , m_bUseUnits(false)
{
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyFloat::prtyFloat(const std::string& i_Name)
  : prtyPropertyTemplate(i_Name, 0.0f)
  , m_bUseUnits(false)
{
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
prtyFloat::prtyFloat(const std::string& i_Name, float i_fInitialValue)
  : prtyPropertyTemplate(i_Name, i_fInitialValue)
  , m_bUseUnits(false)
{
}

//--------------------------------------------------------------------
//	The type of property it is
//--------------------------------------------------------------------
const char*
prtyFloat::GetType()
{
  return "Float";
}

//--------------------------------------------------------------------
// Set whether this property should consider the current units
// when displaying its value in the user interface.
//--------------------------------------------------------------------
bool
prtyFloat::GetUseUnits() const
{
  return m_bUseUnits;
}
void
prtyFloat::SetUseUnits(bool i_bUseUnits)
{
  m_bUseUnits = i_bUseUnits;
}

//--------------------------------------------------------------------
// Get and Set value in the current display units.
// Converts to internal units and then calls Get/SetValue()
//--------------------------------------------------------------------
float
prtyFloat::GetScaledValue() const
{
  if (this->GetUseUnits())
    return (this->GetValue() / prtyUnits::GetUnitScaling());
  else
    return this->GetValue();
}
void
prtyFloat::SetScaledValue(const float i_fValue, bool i_bDirty) // UndoFlags i_Undoable)
{
  if (this->GetUseUnits())
    this->SetValue(i_fValue * prtyUnits::GetUnitScaling(), i_bDirty); // i_Undoable);
  else
    return this->SetValue(i_fValue, i_bDirty); // i_Undoable);
}

//--------------------------------------------------------------------
//	operators
//--------------------------------------------------------------------
prtyFloat&
prtyFloat::operator=(const prtyFloat& i_Property)
{
  // copy base data
  prtyProperty::operator=(i_Property);
  SetValue(i_Property.GetValue());
  m_bUseUnits = i_Property.m_bUseUnits;
  return *this;
}
prtyFloat&
prtyFloat::operator=(const float i_Value)
{
  // bga - This used to just do an operator= on the values, meaning
  //  that notifications were not made. But that is inconsistent with the
  //  other properties.
  SetValue(i_Value);
  return *this;
}

//--------------------------------------------------------------------
//	comparison operators
//--------------------------------------------------------------------
bool
prtyFloat::operator==(const prtyFloat& i_Property) const
{
  return (m_Value == i_Property.GetValue());
}
bool
prtyFloat::operator!=(const prtyFloat& i_Property) const
{
  return (m_Value != i_Property.GetValue());
}
bool
prtyFloat::operator==(const float i_Value) const
{
  return (m_Value == i_Value);
}
bool
prtyFloat::operator!=(const float i_Value) const
{
  return (m_Value != i_Value);
}

bool
prtyFloat::operator>(const float i_Value) const
{
  return (m_Value > i_Value);
}
bool
prtyFloat::operator>=(const float i_Value) const
{
  return (m_Value >= i_Value);
}
bool
prtyFloat::operator<(const float i_Value) const
{
  return (m_Value < i_Value);
}
bool
prtyFloat::operator<=(const float i_Value) const
{
  return (m_Value <= i_Value);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyFloat::Read(chReader& io_Reader)
{
  // float temp;
  // io_Reader.Read(temp);
  // SetValue(temp);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyFloat::Write(docWriter& io_Writer) const
{
  io_Writer.writeFloat32(GetPropertyName(), GetValue());
}
