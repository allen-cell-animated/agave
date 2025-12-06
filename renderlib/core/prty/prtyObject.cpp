#include "core/prty/prtyObject.hpp"

#include "core/prty/prtyProperty.hpp"

#include "core/env/envSTLHelpers.hpp"
#include "core/undo/undoUndoMgr.hpp"

//--------------------------------------------------------------------
//--------------------------------------------------------------------
namespace {
//--------------------------------------------------------------------
// Simplest property reference, used by default - it just keeps a
//	pointer to the property in the assumption that the prtyObject
//	will not be deleted. Bad assumption, but it works in some cases.
// In other cases, the virtual function
//	prtyObject::CreateReferenceForProperty
// will have to be overriden in order to create a true logical
//	reference that handles deletion and recreation.
//--------------------------------------------------------------------
class StaticPropertyReference : public prtyPropertyReference
{
public:
  StaticPropertyReference(prtyProperty* i_Property)
    : m_Property(i_Property)
  {
  }

  virtual prtyProperty* GetProperty() { return m_Property; }

private:
  prtyProperty* m_Property;
};
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
prtyObject::prtyObject()
  : m_bReadOnly(false)
{
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
prtyObject::~prtyObject()
{
  m_PropertiesInfo.DeleteAll();
}

//--------------------------------------------------------------------
// If you pass in a direct pointer, a shared_ptr will be wrap
//	it, meaning that the ownership has passed to this class.
//	If you want to maintain your own copy, pass in a shared_ptr
//	and keep a copy of the shared_ptr yourself.
//--------------------------------------------------------------------
void
prtyObject::AddProperty(prtyPropertyUIInfo* i_pUIInfo)
{
  // When a property is assigned to this object, then this object
  // becomes the reference creator
  i_pUIInfo->AssignReferenceCreator(this);

  m_PropertiesInfo.Add(i_pUIInfo);
}
void
prtyObject::AddProperty(std::shared_ptr<prtyPropertyUIInfo> i_pUIInfo)
{
  // When a property is assigned to this object, then this object
  // becomes the reference creator
  i_pUIInfo->AssignReferenceCreator(this);

  m_PropertiesInfo.Add(i_pUIInfo);
}

//--------------------------------------------------------------------
// Remove the property from this object
//--------------------------------------------------------------------
void
prtyObject::RemoveProperty(std::shared_ptr<prtyPropertyUIInfo> i_pUIInfo)
{
  m_PropertiesInfo.Remove(i_pUIInfo);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
prtyPropertyUIInfoContainer&
prtyObject::GetListContainer()
{
  return m_PropertiesInfo;
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
const PropertyUIIList&
prtyObject::GetList() const
{
  return m_PropertiesInfo.GetList();
}

//------------------------------------------------------------------------
//	Sort the list by PropertyName
//------------------------------------------------------------------------
void
prtyObject::SortListByPropertyName()
{
  m_PropertiesInfo.SortByPropertyName();
}

//------------------------------------------------------------------------
//	Sort the list by Category
//------------------------------------------------------------------------
void
prtyObject::SortListByCategory()
{
  m_PropertiesInfo.SortByCategory();
}

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
const prtyProperty*
prtyObject::GetProperty(const std::string& i_Name)
{
  return m_PropertiesInfo.GetProperty(i_Name);
}

//------------------------------------------------------------------------
//	function to dump properties data to log file
//------------------------------------------------------------------------
void
prtyObject::DebugOutput()
{
  // int i = 0;
  // for (i = 0; i < m_PropertiesInfo.GetNumberOfProperties(); ++i)
  {
    PropertyUIIList& Uiil = m_PropertiesInfo.GetPropertyUIInfoList();

    PropertyUIIList::iterator begin = Uiil.begin();
    PropertyUIIList::iterator it = begin;
    while (it != Uiil.end()) {
      std::shared_ptr<prtyPropertyUIInfo>& pPUII = (*it);
      if (pPUII != 0) {
        int i = 0;
        for (i = 0; i < pPUII->GetNumberOfProperties(); ++i) {
          prtyProperty* pProperty = pPUII->GetProperty(i);
          if (pProperty != 0) {
            //	TODO need a way to get the property value as a string
            //	perhaps a "GetValue()" that returns a string?
            //
            // DBG_LOG2("%02d. %s", i, pProperty->GetPropertyName().c_str());
          }
        }
      }
      ++it;
    }
  }
}

//------------------------------------------------------------------------
// ReadOnly - flag for user interface. If true, do not allow user
//	to alter properties (controls should be read-only).
//------------------------------------------------------------------------
bool
prtyObject::IsReadOnly() const
{
  return m_bReadOnly;
}
void
prtyObject::SetReadOnly(bool i_bValue)
{
  m_bReadOnly = i_bValue;
}

//--------------------------------------------------------------------
//	CreateReferenceForProperty - given a property, create a
//	shared_ptr to a prtyPropertyReference to this property.
//--------------------------------------------------------------------
// virtual
std::shared_ptr<prtyPropertyReference>
prtyObject::CreateReferenceForProperty(prtyProperty& i_Property)
{
  std::shared_ptr<prtyPropertyReference> static_reference(new StaticPropertyReference(&i_Property));
  return static_reference;
}

//--------------------------------------------------------------------
//	CreateUndoForProperty - given a property of this object, create an
//	undo operation that preserves the properties current state.
//	This should be called before the value of the property is changed.
//--------------------------------------------------------------------
// virtual
void
prtyObject::CreateUndoForProperty(prtyProperty& i_Property)
{
  std::shared_ptr<prtyPropertyReference> property_reference = this->CreateReferenceForProperty(i_Property);
  if (property_reference) {
    undoUndoOperation* pUndoOp = i_Property.CreateUndoOperation(property_reference);
    if (pUndoOp) {
      undoUndoMgr::AddOperation(pUndoOp);
    }
  }
}
