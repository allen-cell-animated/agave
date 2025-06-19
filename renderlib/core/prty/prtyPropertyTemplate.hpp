/****************************************************************************\
**	prtyPropertyTemplate.hpp
**
**		Template for properties
**
**
**
\****************************************************************************/
#pragma once
#ifdef PRTY_PROPERTYTEMPLATE_HPP
#error prtyPropertyTemplate.hpp multiply included
#endif
#define PRTY_PROPERTYTEMPLATE_HPP

#ifndef PRTY_PROPERTY_HPP
#include "core/prty/prtyProperty.hpp"
#endif
#ifndef PRTY_UNDOTEMPLATE_HPP
#include "core/prty/prtyUndoTemplate.hpp"
#endif

//============================================================================
//============================================================================
template<typename ValueType, typename ConstRefType>
class prtyPropertyTemplate : public prtyProperty
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyPropertyTemplate(const std::string& i_Name, ConstRefType i_InitialValue)
    : prtyProperty(i_Name)
    , m_Value(i_InitialValue)
  {
  }

  //----------------------------------------------------------------------------
  // Get value of property
  //----------------------------------------------------------------------------
  inline ConstRefType GetValue() const { return m_Value; }

  //----------------------------------------------------------------------------
  // Set value of property. The boolean flag is true
  //	if the change is coming from the user interface and therefore
  //	should mark the document containing the property as dirty.
  //----------------------------------------------------------------------------
  // void SetValue(ConstRefType i_Value, UndoFlags i_Undoable = eNoUndo)
  void SetValue(ConstRefType i_Value, bool i_bDirty = false)
  {
    if (m_Value != i_Value) {
      m_Value = i_Value;
      NotifyCallbacksPropertyChanged(i_bDirty);
      // NotifyCallbacksPropertyChanged(i_Undoable != eNoUndo);
    }
  }

  //--------------------------------------------------------------------
  // Set value of property without notifying the callbacks
  //--------------------------------------------------------------------
  void SetValueWithoutNotify(ConstRefType i_Value) { m_Value = i_Value; }

  //--------------------------------------------------------------------
  // Create and return undo operation of correct type for this property.
  // Ownership passes to the caller.
  //--------------------------------------------------------------------
  undoUndoOperation* CreateUndoOperation(std::shared_ptr<prtyPropertyReference> i_pPropertyRef)
  {
    return new prtyUndoTemplate<prtyPropertyTemplate, ValueType>(i_pPropertyRef, this->GetValue());
  }

protected:
  ValueType m_Value;
};
