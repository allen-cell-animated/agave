/*****************************************************************************\
**	prtyUnits.hpp
**
**		Tracks the units for display of properties realted to distance.
**
**
**
\****************************************************************************/
#ifdef PRTY_UNITS_HPP
#error prtyUnits.hpp multiply included
#endif
#define PRTY_UNITS_HPP

#include <string>

//============================================================================
//	Forward References
//============================================================================
class prtyUnitsInterest;

//============================================================================
//============================================================================
class prtyUnitsInterest
{
public:
  //--------------------------------------------------------------------
  //	UnitsChanged - notification that the units for the
  //	user interface has changed.
  //--------------------------------------------------------------------
  virtual void UnitsChanged() = 0;
};

//============================================================================
//============================================================================
namespace prtyUnits {
//--------------------------------------------------------------------
// Enumeration of unit systems
//--------------------------------------------------------------------
enum UnitTypes
{
  e_Centimeters = 0,
  e_Meters,
  e_Inches,
  e_Feet
};

//--------------------------------------------------------------------
//	UnitSystem - set/get units to use for displaying distances
//	in user interface. This is converted into the UnitScaling
//	that converts from internal units (cm) into display units.
//--------------------------------------------------------------------
void
SetUnitSystem(UnitTypes i_Units);
UnitTypes
GetUnitSystem();

//--------------------------------------------------------------------
// Get conversion scaling to convert from display units to
// internal units (cm). Divide by the scaling to convert from
// internal units to display units.
//--------------------------------------------------------------------
float
GetUnitScaling();

//--------------------------------------------------------------------
// GetShortString - return a short encoding for the current units,
// and a function to get the units using this string.
// examples: "cm", "m", "in", "ft"
//--------------------------------------------------------------------
UnitTypes
GetUnitsByShortString(const std::string& i_Units);
std::string
GetShortString(UnitTypes i_Units);

//--------------------------------------------------------------------
//	RegisterUnitsInterest() - add a Units interest to the system
//--------------------------------------------------------------------
void
RegisterUnitsInterest(prtyUnitsInterest* i_pInterest);

//--------------------------------------------------------------------
//	UnRegisterUnitsInterest() - remove a Units interest from the system.
//
//	Note: this will NOT delete the Units interest.  It is up to the
//	registerer.
//--------------------------------------------------------------------
void
UnRegisterUnitsInterest(prtyUnitsInterest* i_pInterest);

};
