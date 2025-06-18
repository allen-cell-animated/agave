/****************************************************************************\
**	prtyInterest.hpp
**
**
**
\****************************************************************************/
#include "core/prty/prtyInterest.hpp"

//----------------------------------------------------------------------------
// Constructor/Destructor
//----------------------------------------------------------------------------
prtyInterest::prtyInterest(std::string i_Name, void (*i_CallbackFunction)(), bool i_bEditable, bool i_bProcedural)
  : m_Name(i_Name)
  , m_CallbackFunction(i_CallbackFunction)
  , m_bEditable(i_bEditable)
  , m_bProcedural(i_bProcedural)
{
}
prtyInterest::~prtyInterest()
{
  m_CallbackFunction = NULL;
}

//----------------------------------------------------------------------------
// Is this interest editable in a control
//----------------------------------------------------------------------------
bool
prtyInterest::IsEditable()
{
  return m_bEditable;
}
void
prtyInterest::SetEditable(bool i_bEditable)
{
  m_bEditable = i_bEditable;
}

//----------------------------------------------------------------------------
// Is this interest a procedural texture, this will override any pre-existing
// data in the control
//----------------------------------------------------------------------------
bool
prtyInterest::IsProcedural()
{
  return m_bProcedural;
}
void
prtyInterest::SetProcedural(bool i_bProcedural)
{
  m_bProcedural = i_bProcedural;
}

//----------------------------------------------------------------------------
// Get the name of the interest
//----------------------------------------------------------------------------
std::string&
prtyInterest::GetName()
{
  return m_Name;
}
void
prtyInterest::SetName(std::string& i_Name)
{
  m_Name = i_Name;
}

//----------------------------------------------------------------------------
// Execute the interest's callback function
//----------------------------------------------------------------------------
void
prtyInterest::Invoke()
{
  if (m_CallbackFunction) {
    m_CallbackFunction();
  }
}
