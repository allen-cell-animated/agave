/****************************************************************************\
**	prtyUnits.cpp
**
**		see .hpp
**
**
**
\****************************************************************************/
#include "core/prty/prtyUnits.hpp"

#include "Logging.h"
#include "core/env/envSTLHelpers.hpp"

#include <vector>

//============================================================================
//============================================================================
namespace {
std::vector<prtyUnitsInterest*> l_UnitsInterestList;

// Internal units are centimeters, but the default
// display units is meters.
prtyUnits::UnitTypes l_CurrentUnits = prtyUnits::e_Meters;
float l_UnitsScaling = 100.0f;
// prtyUnits::UnitTypes l_CurrentUnits = prtyUnits::e_Centimeters;
// float l_UnitsScaling = 1.0f;
}

//--------------------------------------------------------------------
//	UnitSystem - set/get units to use for displaying distances
//	in user interface. This is converted into the UnitScaling
//	that converts from internal units (cm) into display units.
//--------------------------------------------------------------------
void
prtyUnits::SetUnitSystem(UnitTypes i_Units)
{
  // Set UnitsScaling based on a factor that converts
  // centimeters to the current units
  switch (i_Units) {
    default:
    case e_Centimeters:
      l_UnitsScaling = 1.0f;
      break;
    case e_Meters:
      l_UnitsScaling = 100.0f;
      break;
    case e_Inches:
      l_UnitsScaling = 2.54f;
      break;
    case e_Feet:
      l_UnitsScaling = 2.54f * 12.0f;
      break;
  }

  //	notify scale interests
  std::vector<prtyUnitsInterest*>::iterator it, end = l_UnitsInterestList.end();
  for (it = l_UnitsInterestList.begin(); it != end; ++it) {
    (*it)->UnitsChanged();
  }
}
prtyUnits::UnitTypes
prtyUnits::GetUnitSystem()
{
  return l_CurrentUnits;
}

//--------------------------------------------------------------------
// Get conversion scaling to convert from display units to
// internal units (cm). Divide by the scaling to convert from
// internal units to display units.
//--------------------------------------------------------------------
float
prtyUnits::GetUnitScaling()
{
  return l_UnitsScaling;
}

//--------------------------------------------------------------------
// GetShortString - return a short encoding for the current units,
// and a function to set the units using this string.
// examples: "cm", "m", "in", "ft"
//--------------------------------------------------------------------
prtyUnits::UnitTypes
prtyUnits::GetUnitsByShortString(const std::string& i_Units)
{
  if (i_Units == "m")
    return e_Meters;
  else if (i_Units == "in")
    return e_Inches;
  else if (i_Units == "ft")
    return e_Feet;
  else
    return e_Centimeters;
}
std::string
prtyUnits::GetShortString(UnitTypes i_Units)
{
  switch (i_Units) {
    default:
    case e_Centimeters:
      return "cm";
    case e_Meters:
      return "m";
    case e_Inches:
      return "in";
    case e_Feet:
      return "ft";
  }
}

//--------------------------------------------------------------------
//	RegisterUnitsInterest() - add a Units interest to the system
//--------------------------------------------------------------------
void
prtyUnits::RegisterUnitsInterest(prtyUnitsInterest* i_pInterest)
{
  DBG_ASSERT(i_pInterest != 0, "Cannot register a NULL Scale Interest");
  l_UnitsInterestList.push_back(i_pInterest);
}

//--------------------------------------------------------------------
//	UnRegisterUnitsInterest() - remove a Units interest from the system.
//
//	Note: this will NOT delete the Units interest.  It is up to the
//	registerer.
//--------------------------------------------------------------------
void
prtyUnits::UnRegisterUnitsInterest(prtyUnitsInterest* i_pInterest)
{
  envSTLHelpers::RemoveOneValue(l_UnitsInterestList, i_pInterest);
}
