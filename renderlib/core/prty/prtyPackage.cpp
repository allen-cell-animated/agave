#include "core/prty/prtyPackage.hpp"

#include "core/prty/prtyInterestUtil.hpp"

//----------------------------------------------------------------------------
// initialize prty packages
//----------------------------------------------------------------------------
void
prtyPackage::Init()
{
  prtyInterestUtil::Init();
}

//----------------------------------------------------------------------------
// clean up prty packages
//----------------------------------------------------------------------------
void
prtyPackage::CleanUp()
{
  prtyInterestUtil::DeInit();
}