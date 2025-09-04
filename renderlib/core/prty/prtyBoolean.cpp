#include "core/prty/prtyBoolean.hpp"

// #include "core/ch/chReader.hpp"
// #include "core/ch/chWriter.hpp"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyBoolean::prtyBoolean()
  : prtyPropertyTemplate("Boolean", false)
{
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyBoolean::prtyBoolean(const std::string& i_Name)
  : prtyPropertyTemplate(i_Name, false)
{
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
prtyBoolean::prtyBoolean(const std::string& i_Name, bool i_bInitialValue)
  : prtyPropertyTemplate(i_Name, i_bInitialValue)
{
}

//--------------------------------------------------------------------
//	The type of property it is
//--------------------------------------------------------------------
const char*
prtyBoolean::GetType()
{
  return "Boolean";
}

//--------------------------------------------------------------------
//	operators
//--------------------------------------------------------------------
prtyBoolean&
prtyBoolean::operator=(const prtyBoolean& i_Property)
{
  // copy base data
  prtyProperty::operator=(i_Property);

  SetValue(i_Property.GetValue());
  return *this;
}
prtyBoolean&
prtyBoolean::operator=(const bool i_Value)
{
  SetValue(i_Value);
  return *this;
}

//--------------------------------------------------------------------
//	comparison operators
//--------------------------------------------------------------------
bool
prtyBoolean::operator==(const prtyBoolean& i_Property) const
{
  return (m_Value == i_Property.GetValue());
}
bool
prtyBoolean::operator!=(const prtyBoolean& i_Property) const
{
  return (m_Value != i_Property.GetValue());
}
bool
prtyBoolean::operator==(const bool i_Value) const
{
  return (m_Value == i_Value);
}
bool
prtyBoolean::operator!=(const bool i_Value) const
{
  return (m_Value != i_Value);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyBoolean::Read(chReader& io_Reader)
{
  // bool temp;
  // io_Reader.Read(temp);
  // SetValue(temp);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyBoolean::Write(chWriter& io_Writer) const
{
  // io_Writer.Write(GetValue());
}
