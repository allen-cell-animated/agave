/****************************************************************************\
**	prtyFloat.hpp
**
**		Float property
**
**
**
\****************************************************************************/
#pragma once
#ifdef PRTY_FLOAT_HPP
#error prtyFloat.hpp multiply included
#endif
#define PRTY_FLOAT_HPP

#ifndef PRTY_PROPERTYTEMPLATE_HPP
#include "core/prty/prtyPropertyTemplate.hpp"
#endif

//============================================================================
//============================================================================
class prtyFloat : public prtyPropertyTemplate<float, float>
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyFloat();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyFloat(const std::string& i_Name);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyFloat(const std::string& i_Name, float i_fInitialValue);

  //--------------------------------------------------------------------
  //	The type of property it is
  //--------------------------------------------------------------------
  virtual const char* GetType();

  //--------------------------------------------------------------------
  // Set whether this property should consider the current units
  // when displaying its value in the user interface.
  // prtyFloat has a default value of false.
  //--------------------------------------------------------------------
  bool GetUseUnits() const;
  void SetUseUnits(bool i_bUseUnits);

  //--------------------------------------------------------------------
  // Get and Set value in the current display units.
  // Converts to internal units and then calls Get/SetValue()
  // If this property does not use units, the unscaled value is used.
  //--------------------------------------------------------------------
  float GetScaledValue() const;
  void SetScaledValue(const float i_fValue, bool i_bDirty = false); // UndoFlags i_Undoable = eNoUndo);

  //--------------------------------------------------------------------
  //	operators
  //--------------------------------------------------------------------
  prtyFloat& operator=(const prtyFloat& i_Property);
  prtyFloat& operator=(const float i_Value);

  //--------------------------------------------------------------------
  //	comparison operators
  //--------------------------------------------------------------------
  bool operator==(const prtyFloat& i_Property) const;
  bool operator!=(const prtyFloat& i_Property) const;
  bool operator==(const float i_Value) const;
  bool operator!=(const float i_Value) const;

  //--------------------------------------------------------------------
  //	comparison operators
  //--------------------------------------------------------------------
  bool operator>(const float i_Value) const;
  bool operator>=(const float i_Value) const;
  bool operator<(const float i_Value) const;
  bool operator<=(const float i_Value) const;
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Read(chReader& io_Reader);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Write(chWriter& io_Writer) const;

private:
  bool m_bUseUnits;
};

//============================================================================
// prtyDistance is just a convenience name for a prtyFloat that should
// scale with the current units.
//============================================================================
class prtyDistance : public prtyFloat
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyDistance()
    : prtyFloat()
  {
    SetUseUnits(true);
  }

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyDistance(const std::string& i_Name)
    : prtyFloat(i_Name)
  {
    SetUseUnits(true);
  }

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyDistance(const std::string& i_Name, float i_fInitialValue)
    : prtyFloat(i_Name, i_fInitialValue)
  {
    SetUseUnits(true);
  }

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyDistance& operator=(const float i_Value)
  {
    prtyFloat::operator=(i_Value);
    return *this;
  }
};
