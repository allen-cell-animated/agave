#include "core/prty/prtyInterestUtil.hpp"

#include "core/env/envSTLHelpers.hpp"

//============================================================================
//============================================================================
namespace prtyInterestUtil {
//------------------------------------------------------------------------
//------------------------------------------------------------------------
namespace {
std::vector<prtyInterest*> l_Interests;

} // end anonymous namespace

//------------------------------------------------------------------------
//------------------------------------------------------------------------
void
Init()
{
}

//------------------------------------------------------------------------
//------------------------------------------------------------------------
void
DeInit()
{
  envSTLHelpers::DeleteContainer(l_Interests);
}

//------------------------------------------------------------------------
// Run the callback function for the interest
//------------------------------------------------------------------------
void
InvokeInterest(prtyInterest* i_Interest)
{
  i_Interest->Invoke();
}

//------------------------------------------------------------------------
// add a new interest to our list
//------------------------------------------------------------------------
void
AddInterest(std::string& i_InterestName, void (*i_CallbackFunction)(), bool i_bEditable, bool i_bProcedural)
{
  prtyInterest* new_interest = new prtyInterest(i_InterestName, i_CallbackFunction, i_bEditable, i_bProcedural);
  l_Interests.push_back(new_interest);
}

//------------------------------------------------------------------------
// remove an interest from the list
//------------------------------------------------------------------------
void
RemoveInterest(std::string& i_InterestName)
{
  for (int i = 0; i < l_Interests.size(); i++) {
    if (l_Interests[i]->GetName() == i_InterestName) {
      envSTLHelpers::DeleteOneValue(l_Interests, l_Interests[i]);
    }
  }
}

//------------------------------------------------------------------------
// Retrieve the interest list
//------------------------------------------------------------------------
void
GetInterestList(std::vector<prtyInterest*>& o_Interests)
{
  o_Interests = l_Interests;
}

} // end namespace prtyInterestUtil