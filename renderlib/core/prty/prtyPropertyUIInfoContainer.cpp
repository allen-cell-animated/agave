/****************************************************************************\
**	prtyPropertyUIInfoContainer.cpp
**
**		see .hpp
**
**
**
\****************************************************************************/
#include "core/prty/prtyPropertyUIInfoContainer.hpp"

#include "Logging.h"
#include "core/env/envSTLHelpers.hpp"
#include "core/prty/prtyProperty.hpp"

#include <algorithm>
#include <string>

//============================================================================
//============================================================================
namespace {
struct property_match
{
  property_match(prtyProperty* i_pProperty)
    : m_pProperty(i_pProperty) {};
  inline bool operator()(const std::shared_ptr<prtyPropertyUIInfo>& i_PUII)
  {
    for (int i = 0; i < i_PUII->GetNumberOfProperties(); ++i) {
      if (m_pProperty == i_PUII->GetProperty(i))
        return true;
    }
    return false;
  }
  const prtyProperty* m_pProperty;
};
struct property_name_match
{
  property_name_match(const prtyProperty* i_pProperty)
    : m_pProperty(i_pProperty) {};
  inline bool operator()(const std::shared_ptr<prtyPropertyUIInfo>& i_PUII)
  {
    for (int i = 0; i < i_PUII->GetNumberOfProperties(); ++i) {
      if (m_pProperty->GetPropertyName() == i_PUII->GetProperty(i)->GetPropertyName())
        return true;
    }
    return false;
  }
  const prtyProperty* m_pProperty;
};
struct property_name_match_alt
{
  property_name_match_alt(const std::string& i_Name)
    : m_Name(i_Name) {};
  inline bool operator()(const std::shared_ptr<prtyPropertyUIInfo>& i_PUII)
  {
    for (int i = 0; i < i_PUII->GetNumberOfProperties(); ++i) {
      if (strcmp(m_Name.c_str(), i_PUII->GetProperty(i)->GetPropertyName().c_str()) == 0)
        return true;
    }
    return false;
  }
  const std::string m_Name;
};
struct propertyname_sort
{
  inline bool operator()(std::shared_ptr<prtyPropertyUIInfo>& lhs, std::shared_ptr<prtyPropertyUIInfo>& rhs)
  {
    for (int i = 0; i < lhs->GetNumberOfProperties(); ++i) {
      for (int j = 0; j < rhs->GetNumberOfProperties(); ++j) {
        if (lhs->GetProperty(i)->GetPropertyName() < rhs->GetProperty(j)->GetPropertyName())
          return true;
      }
    }
    return false;
  }
};
struct category_sort
{
  inline bool operator()(std::shared_ptr<prtyPropertyUIInfo>& lhs, std::shared_ptr<prtyPropertyUIInfo>& rhs)
  {
    return (lhs->GetCategory() < rhs->GetCategory());
  }
};
}

//--------------------------------------------------------------------
//	destructor
//--------------------------------------------------------------------
prtyPropertyUIInfoContainer::~prtyPropertyUIInfoContainer()
{
  // Note: shared pointers handle this now
  //	do NOT delete UIInfos since this container may not own them.
  // RemoveAll();
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// void prtyPropertyUIInfoContainer::Add(prtyProperty* i_pProperty,
//									  const std::string& i_Category,
//									  const std::string& i_Description )
//{
//	shared_ptr<prtyPropertyUIInfo> pPUII( new prtyPropertyUIInfo(i_pProperty) );
//	m_PropertyUIInfoList.push_back( pPUII );
//
//	pPUII->SetCategory(i_Category);
//	pPUII->SetDescription(i_Description);
//}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
void
prtyPropertyUIInfoContainer::Add(std::shared_ptr<prtyPropertyUIInfo>& i_pUIInfo)
{
  DBG_ASSERT(i_pUIInfo != 0, "Cannot add a NULL prtyPropertyUIInfo");

  m_PropertyUIInfoList.push_back(i_pUIInfo);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
void
prtyPropertyUIInfoContainer::Add(prtyPropertyUIInfo* i_pUIInfo)
{
  DBG_ASSERT(i_pUIInfo != 0, "Cannot add a NULL prtyPropertyUIInfo");

  m_PropertyUIInfoList.push_back(std::shared_ptr<prtyPropertyUIInfo>(i_pUIInfo));
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
void
prtyPropertyUIInfoContainer::Add(const PropertyUIIList& i_UIInfoList)
{
  PropertyUIIList::const_iterator it, end = i_UIInfoList.end();
  for (it = i_UIInfoList.begin(); it != end; ++it) {
    DBG_ASSERT((*it) != 0, "UIInfo listhas a NULL property in its list");

    // TODO check for duplicates
    m_PropertyUIInfoList.push_back((*it));
  }
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void
prtyPropertyUIInfoContainer::Remove(prtyProperty* i_pProperty)
{
  PropertyUIIList::iterator it =
    std::find_if(m_PropertyUIInfoList.begin(), m_PropertyUIInfoList.end(), property_match(i_pProperty));
  if (it != m_PropertyUIInfoList.end()) {
    m_PropertyUIInfoList.erase(it);
  }
}
void
prtyPropertyUIInfoContainer::Remove(std::shared_ptr<prtyPropertyUIInfo>& i_pUIInfo)
{
  DBG_ASSERT(i_pUIInfo != 0, "Cannot remove a NULL prtyPropertyUIInfo");

  // DBG_LOG("before UII remove = " << m_PropertyUIInfoList.size() << " items" );
  m_PropertyUIInfoList.remove(i_pUIInfo);
  // DBG_LOG("after UII remove = " << m_PropertyUIInfoList.size() << " items" );
}

//--------------------------------------------------------------------
//	Remove all UIInfos from the container, but DO NOT delete them
//--------------------------------------------------------------------
void
prtyPropertyUIInfoContainer::RemoveAll()
{
  m_PropertyUIInfoList.clear();
}

//--------------------------------------------------------------------
//	Delete all UIInfos from the container
//--------------------------------------------------------------------
void
prtyPropertyUIInfoContainer::DeleteAll()
{
  // Note: shared pointers handle this now, no real distinction between
  //  RemoveAll() and DeleteAll()
  m_PropertyUIInfoList.clear();
  // envSTLHelpers::DeleteContainer(m_PropertyUIInfoList);
}

//--------------------------------------------------------------------
//	find if the property is in the list by NAME
//--------------------------------------------------------------------
bool
prtyPropertyUIInfoContainer::IsPropertyInList(const prtyProperty* i_pProperty)
{
  PropertyUIIList::iterator it =
    std::find_if(m_PropertyUIInfoList.begin(), m_PropertyUIInfoList.end(), property_name_match(i_pProperty));
  if (it != m_PropertyUIInfoList.end()) {
    return true;
  }
  return false;
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
const prtyProperty*
prtyPropertyUIInfoContainer::GetProperty(const std::string& i_Name)
{
  PropertyUIIList::iterator it =
    std::find_if(m_PropertyUIInfoList.begin(), m_PropertyUIInfoList.end(), property_name_match_alt(i_Name));
  if (it != m_PropertyUIInfoList.end()) {
    return (*it)->GetProperty(i_Name);
  }
  return NULL;
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
const std::shared_ptr<prtyPropertyUIInfo>
prtyPropertyUIInfoContainer::GetPropertyUIInfo(const std::string& i_Name)
{
  PropertyUIIList::iterator it =
    std::find_if(m_PropertyUIInfoList.begin(), m_PropertyUIInfoList.end(), property_name_match_alt(i_Name));
  if (it != m_PropertyUIInfoList.end()) {
    return (*it);
  }
  return std::shared_ptr<prtyPropertyUIInfo>();
}

//--------------------------------------------------------------------
//	get the UIInfo list
//--------------------------------------------------------------------
PropertyUIIList&
prtyPropertyUIInfoContainer::GetPropertyUIInfoList()
{
  return m_PropertyUIInfoList;
}

//------------------------------------------------------------------------
//	Sort the list by Category
//------------------------------------------------------------------------
void
prtyPropertyUIInfoContainer::SortByCategory()
{
  // Sort the vector using predicate and std::sort
  // std::sort(m_PropertyUIInfoList.begin(), m_PropertyUIInfoList.end(), category_sort());
  m_PropertyUIInfoList.sort(category_sort());
}

//------------------------------------------------------------------------
//	Sort the list by PropertyName
//------------------------------------------------------------------------
void
prtyPropertyUIInfoContainer::SortByPropertyName()
{
  // Sort the vector using predicate and std::sort
  // std::sort(m_PropertyUIInfoList.begin(), m_PropertyUIInfoList.end(), propertyname_sort());
  m_PropertyUIInfoList.sort(propertyname_sort());
}

//------------------------------------------------------------------------
// Subcategories allow one property object to have a hierarchy of
// properties where unique names are only needed within each subcategory.
//------------------------------------------------------------------------
int
prtyPropertyUIInfoContainer::GetNumSubCategories() const
{
  return m_SubCategories.size();
}
const prtyPropertyUIInfoContainer&
prtyPropertyUIInfoContainer::GetSubCategory(int i_Index) const
{
  DBG_ASSERT(m_SubCategories[i_Index].m_Container, "Null property subcategory");
  return (*m_SubCategories[i_Index].m_Container);
}

//------------------------------------------------------------------------
// Add subcategory with given set of properties
//------------------------------------------------------------------------
void
prtyPropertyUIInfoContainer::AddSubCategory(const std::string& i_CategoryName,
                                            const std::shared_ptr<prtyPropertyUIInfoContainer>& i_Properties)
{
  sSubCategory subcat = { i_CategoryName, i_Properties };
  m_SubCategories.push_back(subcat);
}

//------------------------------------------------------------------------
// Remove subcategory
//------------------------------------------------------------------------
void
prtyPropertyUIInfoContainer::RemoveSubCategory(const std::shared_ptr<prtyPropertyUIInfoContainer>& i_Properties)
{
  std::vector<sSubCategory>::iterator it;
  for (it = m_SubCategories.begin(); it != m_SubCategories.end(); ++it) {
    if (it->m_Container == i_Properties)
      break;
  }

  if (it != m_SubCategories.end())
    m_SubCategories.erase(it);
}

//------------------------------------------------------------------------
// There are two types of sub category name functions. One is the internal
// name for finding sub categories for python and scripting. The other
// is the display name, which can be localized.
//------------------------------------------------------------------------
const std::string&
prtyPropertyUIInfoContainer::GetSubCategoryName(int i_Index) const
{
  return m_SubCategories[i_Index].m_Name;
}
const std::string&
prtyPropertyUIInfoContainer::GetSubCategoryDisplayName(int i_Index) const
{
  // No localization yet
  return m_SubCategories[i_Index].m_Name;
}

//------------------------------------------------------------------------
// Set/Get show subcategory flag
//------------------------------------------------------------------------
void
prtyPropertyUIInfoContainer::SetShowSubCategory(bool i_bShow)
{
  m_bShowSubCategories = i_bShow;
}

bool
prtyPropertyUIInfoContainer::GetShowSubCategory() const
{
  return m_bShowSubCategories;
}