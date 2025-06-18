/****************************************************************************\
**	prtyPropertyUIInfoContainer.hpp
**
**		Container for property UI info
**
**
**
\****************************************************************************/
#pragma once
#ifdef PRTY_PROPERTYUIINFOCONTAINER_HPP
#error prtyPropertyUIInfoContainer.hpp multiply included
#endif
#define PRTY_PROPERTYUIINFOCONTAINER_HPP

#ifndef PRTY_PROPERTYUIINFO_HPP
#include "core/prty/prtyPropertyUIInfo.hpp"
#endif

#include <list>

//============================================================================
//============================================================================
class prtyProperty;
class nameString;

//============================================================================
//============================================================================
// typedef std::list<prtyPropertyUIInfo*> PropertyUIIList;

// Using shared pointer here allows multiple views of the same properties,
// and allows the controls to maintain weak poitners to the ui info
// in order to know when the property info is no longer valid.
typedef std::list<std::shared_ptr<prtyPropertyUIInfo>> PropertyUIIList;

//============================================================================
//============================================================================
class prtyPropertyUIInfoContainer
{
public:
  //--------------------------------------------------------------------
  //	destructor - this calls OWNS the prtyPropertyUIInfo instantiations
  //--------------------------------------------------------------------
  ~prtyPropertyUIInfoContainer();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  // void Add(prtyProperty* i_pProperty,
  //		const std::string& i_Category,
  //		const std::string& i_Description);

  //--------------------------------------------------------------------
  // Internally, the property UI Info list uses shared_ptr, so
  //	if you want to maintain a copy yourself, pass in a shared
  //	pointer. Otherwise, ownership of the pointer will be
  //	controlled by this list.
  //--------------------------------------------------------------------
  void Add(prtyPropertyUIInfo* i_pUIInfo);
  void Add(std::shared_ptr<prtyPropertyUIInfo>& i_pUIInfo);
  void Add(const PropertyUIIList& i_UIInfoList);

  //--------------------------------------------------------------------
  // If you want to remove a UIInfo, keep a copy of the shared_ptr
  //	around and use that to remove it.
  //--------------------------------------------------------------------
  void Remove(prtyProperty* i_pProperty);
  void Remove(std::shared_ptr<prtyPropertyUIInfo>& i_pUIInfo);

  //--------------------------------------------------------------------
  //	Remove all UIInfos from the container, but DO NOT delete them
  //		Note: this changes now that the internals are share_ptr.
  //			There is no more difference between RemoveAll() and
  //			DeleteAll().
  //--------------------------------------------------------------------
  void RemoveAll();

  //--------------------------------------------------------------------
  //	Delete all UIInfos from the container
  //--------------------------------------------------------------------
  void DeleteAll();

  //--------------------------------------------------------------------
  //	find if the property is in the list by NAME
  //--------------------------------------------------------------------
  bool IsPropertyInList(const prtyProperty* i_pProperty);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  inline const PropertyUIIList& GetList() const { return m_PropertyUIInfoList; }

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  const prtyProperty* GetProperty(const std::string& i_Name);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  const std::shared_ptr<prtyPropertyUIInfo> GetPropertyUIInfo(const std::string& i_Name);

  //--------------------------------------------------------------------
  //	get the UIInfo list
  //--------------------------------------------------------------------
  PropertyUIIList& GetPropertyUIInfoList();

  //------------------------------------------------------------------------
  //	Sort the list by PropertyName
  //------------------------------------------------------------------------
  void SortByPropertyName();

  //------------------------------------------------------------------------
  //	Sort the list by Category
  //------------------------------------------------------------------------
  void SortByCategory();

  //------------------------------------------------------------------------
  // Subcategories allow one property object to have a hierarchy of
  // properties where unique names are only needed within each subcategory.
  //------------------------------------------------------------------------
  int GetNumSubCategories() const;
  const prtyPropertyUIInfoContainer& GetSubCategory(int i_Index) const;

  //------------------------------------------------------------------------
  // Add subcategory with given set of properties
  //------------------------------------------------------------------------
  void AddSubCategory(const std::string& i_CategoryName,
                      const std::shared_ptr<prtyPropertyUIInfoContainer>& i_Properties);

  //------------------------------------------------------------------------
  // Remove subcategory
  //------------------------------------------------------------------------
  void RemoveSubCategory(const std::shared_ptr<prtyPropertyUIInfoContainer>& i_Properties);

  //------------------------------------------------------------------------
  // There are two types of sub category name functions. One is the internal
  // name for finding sub categories for python and scripting. The other
  // is the display name, which can be localized.
  //------------------------------------------------------------------------
  const std::string& GetSubCategoryName(int i_Index) const;
  const std::string& GetSubCategoryDisplayName(int i_Index) const;

  //------------------------------------------------------------------------
  // Set/Get show subcategory flag
  //------------------------------------------------------------------------
  void SetShowSubCategory(bool i_bShow);
  bool GetShowSubCategory() const;

private:
  PropertyUIIList m_PropertyUIInfoList;

  struct sSubCategory
  {
    std::string m_Name;
    std::shared_ptr<prtyPropertyUIInfoContainer> m_Container;
  };

  std::vector<sSubCategory> m_SubCategories;

  bool m_bShowSubCategories;
};
