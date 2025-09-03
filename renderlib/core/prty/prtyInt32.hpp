/****************************************************************************\
**	prtyInt32.hpp
**
**		Int32 property
**
**
**
\****************************************************************************/
#ifdef PRTY_INT32_HPP
#error prtyInt32.hpp multiply included
#endif
#define PRTY_INT32_HPP

#ifndef PRTY_PROPERTYTEMPLATE_HPP
#include "core/prty/prtyPropertyTemplate.hpp"
#endif

//============================================================================
//============================================================================
class prtyInt32 : public prtyPropertyTemplate<int, int>
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyInt32();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyInt32(const std::string& i_Name);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyInt32(const std::string& i_Name, const int& i_InitialValue);

  //--------------------------------------------------------------------
  //	The type of property it is
  //--------------------------------------------------------------------
  virtual const char* GetType();

  //--------------------------------------------------------------------
  //	operators
  //--------------------------------------------------------------------
  prtyInt32& operator=(const prtyInt32& i_Property);
  prtyInt32& operator=(const int i_Value);

  //--------------------------------------------------------------------
  //	comparison operators
  //--------------------------------------------------------------------
  bool operator==(const prtyInt32& i_Property) const;
  bool operator!=(const prtyInt32& i_Property) const;
  bool operator==(const int i_Value) const;
  bool operator!=(const int i_Value) const;

  //--------------------------------------------------------------------
  //	comparison operators
  //--------------------------------------------------------------------
  bool operator>(const int i_Value) const;
  bool operator>=(const int i_Value) const;
  bool operator<(const int i_Value) const;
  bool operator<=(const int i_Value) const;
  bool operator>(const prtyInt32& i_Value) const;
  bool operator>=(const prtyInt32& i_Value) const;
  bool operator<(const prtyInt32& i_Value) const;
  bool operator<=(const prtyInt32& i_Value) const;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Read(chReader& io_Reader);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Write(chWriter& io_Writer) const;
};
