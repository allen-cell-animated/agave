#include "core/prty/prtyColor.hpp"

// #include "core/ch/chChunkParserUtil.hpp"
// #include "core/ch/chReader.hpp"
#include "serialize/docWriter.h"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyColor::prtyColor()
  : prtyPropertyTemplate("Color", glm::vec4(0, 0, 0, 0))
{
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyColor::prtyColor(const std::string& i_Name)
  : prtyPropertyTemplate(i_Name, glm::vec4(0, 0, 0, 0))
{
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyColor::prtyColor(const std::string& i_Name, const glm::vec4& i_InitialValue)
  : prtyPropertyTemplate(i_Name, i_InitialValue)
{
}

//--------------------------------------------------------------------
//	The type of property it is
//--------------------------------------------------------------------
const char*
prtyColor::GetType()
{
  return "Color";
}

//--------------------------------------------------------------------
//	operators
//--------------------------------------------------------------------
prtyColor&
prtyColor::operator=(const prtyColor& i_Property)
{
  // copy base data
  prtyProperty::operator=(i_Property);

  SetValue(i_Property.GetValue());
  return *this;
}
prtyColor&
prtyColor::operator=(const glm::vec4& i_Value)
{
  SetValue(i_Value);
  return *this;
}

//--------------------------------------------------------------------
//	comparison operators
//--------------------------------------------------------------------
bool
prtyColor::operator==(const prtyColor& i_Property) const
{
  return (m_Value == i_Property.GetValue());
}
bool
prtyColor::operator!=(const prtyColor& i_Property) const
{
  return (m_Value != i_Property.GetValue());
}
bool
prtyColor::operator==(const glm::vec4& i_Value) const
{
  return (m_Value == i_Value);
}
bool
prtyColor::operator!=(const glm::vec4& i_Value) const
{
  return (m_Value != i_Value);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyColor::Read(chReader& io_Reader)
{
  // glm::vec4 temp;
  // chChunkParserUtil::Read(io_Reader, temp);
  // SetValue(temp);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyColor::Write(docWriter& io_Writer) const
{
  io_Writer.writeFloat32Array(GetPropertyName(), 4, glm::value_ptr(GetValue()));
}
