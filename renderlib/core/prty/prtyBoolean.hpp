/****************************************************************************\
**	prtyBoolean.hpp
**
**		Boolean property
**
**
**
\****************************************************************************/
#pragma once
#ifdef PRTY_BOOLEAN_HPP
#error prtyBoolean.hpp multiply included
#endif
#define PRTY_BOOLEAN_HPP

#ifndef PRTY_PROPERTYTEMPLATE_HPP
#include "core/prty/prtyPropertyTemplate.hpp"
#endif

//============================================================================
//============================================================================
class prtyBoolean : public prtyPropertyTemplate<bool, bool>
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyBoolean();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyBoolean(const std::string& i_Name);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyBoolean(const std::string& i_Name, bool i_bInitialValue);

  //--------------------------------------------------------------------
  //	The type of property it is
  //--------------------------------------------------------------------
  virtual const char* GetType();

  //--------------------------------------------------------------------
  //	operators
  //--------------------------------------------------------------------
  prtyBoolean& operator=(const prtyBoolean& i_Property);
  prtyBoolean& operator=(const bool i_Value);

  //--------------------------------------------------------------------
  //	comparison operators
  //--------------------------------------------------------------------
  bool operator==(const prtyBoolean& i_Property) const;
  bool operator!=(const prtyBoolean& i_Property) const;
  bool operator==(const bool i_Value) const;
  bool operator!=(const bool i_Value) const;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Read(chReader& io_Reader);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Write(chWriter& io_Writer) const;
};
