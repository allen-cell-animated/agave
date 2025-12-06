#include "core/prty/prtyEnum.hpp"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyEnum::prtyEnum()
  : prtyInt8("Enum")
{
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyEnum::prtyEnum(const std::string& i_Name)
  : prtyInt8(i_Name)
{
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyEnum::prtyEnum(const std::string& i_Name, const char& i_InitialValue)
  : prtyInt8(i_Name, i_InitialValue)
{
}

//--------------------------------------------------------------------
//	The type of property it is
//--------------------------------------------------------------------
const char*
prtyEnum::GetType()
{
  return "Enum";
}

//--------------------------------------------------------------------
//	Set an EnumTag	- these will get displayed (in a combobox for
//	instance)
//--------------------------------------------------------------------
void
prtyEnum::SetEnumTag(int i_Index, const std::string& i_Tag)
{
  if (i_Index >= (int)m_EnumTags.size()) {
    m_EnumTags.resize(i_Index + 1);
  }
  m_EnumTags[i_Index] = i_Tag;

  // int numtags = m_EnumTags.size();
  // this->SetMaximum(numtags);
}

const std::string&
prtyEnum::GetEnumTag(int i_Index)
{
  return m_EnumTags[i_Index];
}

//--------------------------------------------------------------------
// Return number of enumeration tags
//--------------------------------------------------------------------
int
prtyEnum::GetNumTags() const
{
  return m_EnumTags.size();
}

//--------------------------------------------------------------------
// Clear enum tags
//--------------------------------------------------------------------
void
prtyEnum::ClearTags()
{
  m_EnumTags.clear();
}