/****************************************************************************\
**	prtyColorRGBAEditUIInfo.hpp
**
**		ColorRGBAEdit UI info
**
**
**
\****************************************************************************/
#ifdef PRTY_COLORRGBAEDITUIINFO_HPP
#error prtyColorRGBAEditUIInfo.hpp multiply included
#endif
#define PRTY_COLORRGBAEDITUIINFO_HPP

#ifndef PRTY_PROPERTYUIINFO_HPP
#include "core/prty/prtyPropertyUIInfo.hpp"
#endif

#include <string>

//============================================================================
//============================================================================
class prtyProperty;

//============================================================================
//============================================================================
class prtyColorRGBAEditUIInfo : public prtyPropertyUIInfo
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyColorRGBAEditUIInfo(prtyProperty* i_pProperty);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyColorRGBAEditUIInfo(prtyProperty* i_pProperty, const std::string& i_Category, const std::string& i_Description);

  //--------------------------------------------------------------------
  // Return pointer to new equivalent prtyPropertyUIInfo
  //--------------------------------------------------------------------
  virtual prtyPropertyUIInfo* Clone();
};
