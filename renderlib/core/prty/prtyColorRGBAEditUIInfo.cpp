/****************************************************************************\
**	prtyColorRGBAEditUIInfo.cpp
**
**		see .hpp
**
**
**
\****************************************************************************/
#include "core/prty/prtyColorRGBAEditUIInfo.hpp"

#include "core/prty/prtyProperty.hpp"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyColorRGBAEditUIInfo::prtyColorRGBAEditUIInfo(prtyProperty* i_pProperty)
  : prtyPropertyUIInfo(i_pProperty)
{
  SetControlName("ColorRGBAEdit");
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyColorRGBAEditUIInfo::prtyColorRGBAEditUIInfo(prtyProperty* i_pProperty,
                                                 const std::string& i_Category,
                                                 const std::string& i_Description)
  : prtyPropertyUIInfo(i_pProperty, i_Category, i_Description)
{
  SetControlName("ColorRGBAEdit");
}

//--------------------------------------------------------------------
// Return pointer to new equivalent prtyPropertyUIInfo
//--------------------------------------------------------------------
// virtual
prtyPropertyUIInfo*
prtyColorRGBAEditUIInfo::Clone()
{
  return new prtyColorRGBAEditUIInfo(this->GetProperty(0));
}
