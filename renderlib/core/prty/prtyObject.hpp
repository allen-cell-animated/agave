/****************************************************************************\
**	prtyObject.hpp
**
**		Object base class
**
**
**
\****************************************************************************/
#pragma once
#ifdef PRTY_OBJECT_HPP
#error prtyObject.hpp multiply included
#endif
#define PRTY_OBJECT_HPP

#ifndef PRTY_PROPERTYUIINFO_HPP
#include "core/prty/prtyPropertyUIInfo.hpp"
#endif
#ifndef PRTY_PROPERTYUIINFOCONTAINER_HPP
#include "core/prty/prtyPropertyUIInfoContainer.hpp"
#endif
#ifndef PRTY_PROPERTYREFERENCE_HPP
#include "core/prty/prtyPropertyReference.hpp"
#endif

#include <string>

//============================================================================
//============================================================================
class prtyObject : public prtyReferenceCreator
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyObject();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual ~prtyObject();

  //--------------------------------------------------------------------
  // If you pass in a direct pointer, a shared_ptr will be wrap
  //	it, meaning that the ownership has passed to this class.
  //	If you want to maintain your own copy, pass in a shared_ptr
  //	and keep a copy of the shared_ptr yourself.
  //--------------------------------------------------------------------
  void AddProperty(prtyPropertyUIInfo* i_pUIInfo);
  void AddProperty(std::shared_ptr<prtyPropertyUIInfo> i_pUIInfo);

  //--------------------------------------------------------------------
  // Remove the property from this object
  //--------------------------------------------------------------------
  void RemoveProperty(std::shared_ptr<prtyPropertyUIInfo> i_pUIInfo);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyPropertyUIInfoContainer& GetListContainer();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  const PropertyUIIList& GetList() const;

  //------------------------------------------------------------------------
  //	Sort the list by PropertyName
  //------------------------------------------------------------------------
  void SortListByPropertyName();

  //------------------------------------------------------------------------
  //	Sort the list by Category
  //------------------------------------------------------------------------
  void SortListByCategory();

  //------------------------------------------------------------------------
  //------------------------------------------------------------------------
  const prtyProperty* GetProperty(const std::string& i_Name);

  //------------------------------------------------------------------------
  //	function to dump properties data to log file
  //------------------------------------------------------------------------
  void DebugOutput();

  //------------------------------------------------------------------------
  // ReadOnly - flag for user interface. If true, do not allow user
  //	to alter properties (controls should be read-only).
  //------------------------------------------------------------------------
  bool IsReadOnly() const;
  void SetReadOnly(bool i_bValue);

  //--------------------------------------------------------------------
  //	CreateReferenceForProperty - given a property, create a
  //	shared_ptr to a prtyPropertyReference to this property.
  //--------------------------------------------------------------------
  virtual std::shared_ptr<prtyPropertyReference> CreateReferenceForProperty(prtyProperty& i_Property);

  //--------------------------------------------------------------------
  //	CreateUndoForProperty - given a property of this object, create an
  //	undo operation that preserves the properties current state.
  //	This should be called before the value of the property is changed.
  //--------------------------------------------------------------------
  virtual void CreateUndoForProperty(prtyProperty& i_Property);

private:
  prtyPropertyUIInfoContainer m_PropertiesInfo;
  bool m_bReadOnly;
};
