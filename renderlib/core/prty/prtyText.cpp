/****************************************************************************\
**	prtyText.cpp
**
**		see .hpp
**
**
**
\****************************************************************************/
#include "core/prty/prtyText.hpp"

// #include "core/ch/chChunkParserUtil.hpp"
// #include "core/ch/chReader.hpp"

//--------------------------------------------------------------------
//--------------------------------------------------------------------
prtyText::prtyText()
  : prtyPropertyTemplate("Text", std::string())
{
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
prtyText::prtyText(const std::string& i_Name)
  : prtyPropertyTemplate(i_Name, std::string())
{
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
prtyText::prtyText(const std::string& i_Name, const std::string& i_InitialValue)
  : prtyPropertyTemplate(i_Name, i_InitialValue)
{
}

//--------------------------------------------------------------------
//	The type of property it is
//--------------------------------------------------------------------
const char*
prtyText::GetType()
{
  return "Text";
}

//--------------------------------------------------------------------
//	operators
//--------------------------------------------------------------------
prtyText&
prtyText::operator=(const prtyText& i_Property)
{
  // copy base data
  prtyProperty::operator=(i_Property);

  SetValue(i_Property.GetValue());
  return *this;
}
prtyText&
prtyText::operator=(const std::string& i_Value)
{
  SetValue(i_Value);
  return *this;
}
prtyText&
prtyText::operator=(const char* i_Value)
{
  SetValue(std::string(i_Value));
  return *this;
}

//--------------------------------------------------------------------
//	comparison operators
//--------------------------------------------------------------------
bool
prtyText::operator==(const prtyText& i_Property) const
{
  return (m_Value == i_Property.GetValue());
}
bool
prtyText::operator!=(const prtyText& i_Property) const
{
  return (m_Value != i_Property.GetValue());
}
bool
prtyText::operator==(const std::string& i_Value) const
{
  return (m_Value == i_Value);
}
bool
prtyText::operator!=(const std::string& i_Value) const
{
  return (m_Value != i_Value);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyText::Read(chReader& io_Reader)
{
  // std::string temp;
  // chChunkParserUtil::Read(io_Reader, temp);
  // SetValue(temp);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyText::Write(chWriter& io_Writer) const
{
  //  chChunkParserUtil::Write(io_Writer, GetValue());
}
