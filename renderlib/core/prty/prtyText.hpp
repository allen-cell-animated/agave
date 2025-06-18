/****************************************************************************\
**	prtyText.hpp
**
**		Text property
**
**
**
\****************************************************************************/
#ifdef PRTY_TEXT_HPP
#error prtyText.hpp multiply included
#endif
#define PRTY_TEXT_HPP

#ifndef PRTY_PROPERTYTEMPLATE_HPP
#include "core/prty/prtyPropertyTemplate.hpp"
#endif

//============================================================================
//============================================================================
class prtyText : public prtyPropertyTemplate<std::string, const std::string&>
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyText();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyText(const std::string& i_Name);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyText(const std::string& i_Name, const std::string& i_InitialValue);

  //--------------------------------------------------------------------
  //	The type of property it is
  //--------------------------------------------------------------------
  virtual const char* GetType();

  //--------------------------------------------------------------------
  //	operators
  //--------------------------------------------------------------------
  prtyText& operator=(const prtyText& i_Property);
  prtyText& operator=(const std::string& i_Value);
  prtyText& operator=(const char* i_Value);

  //--------------------------------------------------------------------
  //	comparison operators
  //--------------------------------------------------------------------
  bool operator==(const prtyText& i_Property) const;
  bool operator!=(const prtyText& i_Property) const;
  bool operator==(const std::string& i_Value) const;
  bool operator!=(const std::string& i_Value) const;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Read(chReader& io_Reader);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Write(chWriter& io_Writer) const;
};
