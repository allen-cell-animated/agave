/****************************************************************************\
**	prtyPropertyUIInfo.cpp
**
**		see .hpp
**
**
**
\****************************************************************************/
#include "core/prty/prtyPropertyUIInfo.hpp"

#include "core/prty/prtyProperty.hpp"

#include "Logging.h"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyPropertyUIInfo::prtyPropertyUIInfo(prtyProperty* i_pProperty)
  : m_Category("")
  , m_Description("")
  , m_bReadOnly(false)
{
  if (i_pProperty != 0) {
    AddProperty(i_pProperty);
  }
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyPropertyUIInfo::prtyPropertyUIInfo(prtyProperty* i_pProperty,
                                       const std::string& i_Category,
                                       const std::string& i_Description)
  : m_Category(i_Category)
  , m_Description(i_Description)
  , m_bReadOnly(false)
{
  if (i_pProperty != 0) {
    AddProperty(i_pProperty);
  }
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
prtyPropertyUIInfo::~prtyPropertyUIInfo() {}

//--------------------------------------------------------------------
// Return pointer to new equivalent prtyPropertyUIInfo
//--------------------------------------------------------------------
// virtual
prtyPropertyUIInfo*
prtyPropertyUIInfo::Clone()
{
  return new prtyPropertyUIInfo(this->GetProperty(0));
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
int
prtyPropertyUIInfo::GetNumberOfProperties() const
{
  return m_Properties.size();
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyProperty*
prtyPropertyUIInfo::GetProperty(int i_Index) const
{
  DBG_ASSERT((i_Index >= 0) && (i_Index < m_Properties.size()), "Invalid property list index");

  return m_Properties[i_Index];
}
prtyProperty*
prtyPropertyUIInfo::GetProperty(const std::string& i_Name) const
{
  for (int i = 0; i < m_Properties.size(); ++i) {
    if (strcmp(i_Name.c_str(), m_Properties[i]->GetPropertyName().c_str()) == 0)
      return m_Properties[i];
  }
  return 0;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyReferenceCreator*
prtyPropertyUIInfo::GetReferenceCreator(int i_Index) const
{
  DBG_ASSERT((i_Index >= 0) && (i_Index < m_ReferenceCreators.size()), "Invalid reference creator list index");
  return m_ReferenceCreators[i_Index];
}

//--------------------------------------------------------------------
// Add property to list, optionally passing a reference creator
// that can be used when creating undo operations.
//--------------------------------------------------------------------
void
prtyPropertyUIInfo::AddProperty(prtyProperty* i_pProperty, prtyReferenceCreator* i_pReferenceCreator)
{
  m_Properties.push_back(i_pProperty);
  m_ReferenceCreators.push_back(i_pReferenceCreator);

  // DBG_LOG2("=UIInfo add property (%s) count = %d", this->GetControlName().c_str(), m_Properties.size() );
}

//--------------------------------------------------------------------
// Can be used to assign a new reference creator to all existing
//	properties in this UIInfo. This will not affect new properties
//	added after this call.
//--------------------------------------------------------------------
void
prtyPropertyUIInfo::AssignReferenceCreator(prtyReferenceCreator* i_pReferenceCreator)
{
  for (int i = 0; i < m_ReferenceCreators.size(); ++i)
    m_ReferenceCreators[i] = i_pReferenceCreator;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
const std::string&
prtyPropertyUIInfo::GetControlName() const
{
  return m_ControlName;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
const std::string&
prtyPropertyUIInfo::GetCategory() const
{
  return m_Category;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void
prtyPropertyUIInfo::SetCategory(const std::string& i_Category)
{
  m_Category = i_Category;
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
const std::string&
prtyPropertyUIInfo::GetDescription() const
{
  return m_Description;
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
void
prtyPropertyUIInfo::SetDescription(const std::string& i_Description)
{
  m_Description = i_Description;
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
const bool
prtyPropertyUIInfo::GetReadOnly() const
{
  return m_bReadOnly;
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
void
prtyPropertyUIInfo::SetReadOnly(const bool i_bReadOnly)
{
  m_bReadOnly = i_bReadOnly;
}

//--------------------------------------------------------------------
// ConfirmationString is message to display to user that he needs
// to confirm before the value of the property can be changed.
//--------------------------------------------------------------------
void
prtyPropertyUIInfo::SetConfirmationString(const std::string& i_Confirmation)
{
  m_ConfirmationString = i_Confirmation;
}
const std::string&
prtyPropertyUIInfo::GetConfirmationString() const
{
  return m_ConfirmationString;
}

//--------------------------------------------------------------------
//	Call this function when the property UI Info has been changed
//	after a control for it has already been created. It will
//	cause the control to be updated.
//	Note: Con't call this automatically from derived classes,
//	just let the prtyObject that owns this UI Info handle it.
//--------------------------------------------------------------------
void
prtyPropertyUIInfo::UpdateControl()
{
  // prtyIntCallbackMgr::UpdateControl(this);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
void
prtyPropertyUIInfo::SetControlName(const std::string i_ControlName)
{
  m_ControlName = i_ControlName;
}
