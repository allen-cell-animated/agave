#pragma once

#include "core/prty/prtyPropertyUIInfo.hpp"

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
