/****************************************************************************\
**	prtyColor.hpp
**
**		Color property
**
**
**
\****************************************************************************/
#pragma once
#ifdef PRTY_COLOR_HPP
#error prtyColor.hpp multiply included
#endif
#define PRTY_COLOR_HPP

#ifndef PRTY_PROPERTYTEMPLATE_HPP
#include "core/prty/prtyPropertyTemplate.hpp"
#endif

#include "glm.h"

//============================================================================
//============================================================================
class prtyColor : public prtyPropertyTemplate<glm::vec4, const glm::vec4&>
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyColor();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyColor(const std::string& i_Name);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyColor(const std::string& i_Name, const glm::vec4& i_InitialValue);

  //--------------------------------------------------------------------
  //	The type of property it is
  //--------------------------------------------------------------------
  virtual const char* GetType();

  //--------------------------------------------------------------------
  //	operators
  //--------------------------------------------------------------------
  prtyColor& operator=(const prtyColor& i_Property);
  prtyColor& operator=(const glm::vec4& i_Value);

  //--------------------------------------------------------------------
  //	comparison operators
  //--------------------------------------------------------------------
  bool operator==(const prtyColor& i_Property) const;
  bool operator!=(const prtyColor& i_Property) const;
  bool operator==(const glm::vec4& i_Value) const;
  bool operator!=(const glm::vec4& i_Value) const;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Read(chReader& io_Reader);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Write(chWriter& io_Writer) const;
};
