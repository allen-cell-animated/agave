#pragma once

#include "core/prty/prtyPropertyTemplate.hpp"

//============================================================================
//============================================================================
class prtyInt8 : public prtyPropertyTemplate<int8_t, int8_t>
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyInt8();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyInt8(const std::string& i_Name);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyInt8(const std::string& i_Name, int8_t i_InitialValue);

  //--------------------------------------------------------------------
  //	The type of property it is
  //--------------------------------------------------------------------
  virtual const char* GetType();

  //--------------------------------------------------------------------
  //	operators
  //--------------------------------------------------------------------
  prtyInt8& operator=(const prtyInt8& i_Property);
  prtyInt8& operator=(const int8_t i_Value);

  //--------------------------------------------------------------------
  //	comparison operators
  //--------------------------------------------------------------------
  bool operator==(const prtyInt8& i_Property) const;
  bool operator!=(const prtyInt8& i_Property) const;
  bool operator==(const int8_t i_Value) const;
  bool operator!=(const int8_t i_Value) const;

  //--------------------------------------------------------------------
  //	comparison operators
  //--------------------------------------------------------------------
  bool operator>(const int8_t i_Value) const;
  bool operator>=(const int8_t i_Value) const;
  bool operator<(const int8_t i_Value) const;
  bool operator<=(const int8_t i_Value) const;
  bool operator>(const prtyInt8& i_Value) const;
  bool operator>=(const prtyInt8& i_Value) const;
  bool operator<(const prtyInt8& i_Value) const;
  bool operator<=(const prtyInt8& i_Value) const;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Read(docReader& io_Reader);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Write(docWriter& io_Writer) const;
};
