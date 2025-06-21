/****************************************************************************\
**	prtyPropertyTemplate.hpp
**
**		Template for properties
**
**
**
\****************************************************************************/
#pragma once

#include "core/prty/prtyProperty.hpp"
#include "core/prty/prtyUndoTemplate.hpp"

#include <iostream>

//============================================================================
//============================================================================
template<typename ValueType,
         typename = std::enable_if_t<
           std::is_copy_constructible_v<ValueType> && std::is_copy_assignable_v<ValueType> &&
           std::is_default_constructible_v<ValueType> && std::is_move_constructible_v<ValueType> &&
           std::is_move_assignable_v<ValueType> &&
           std::is_convertible_v<decltype(std::declval<std::ostream&>() << std::declval<ValueType>()), std::ostream&>>>
class prtyPropertyTemplate : public prtyProperty
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyPropertyTemplate(const std::string& i_Name, const ValueType& i_InitialValue)
    : prtyProperty(i_Name)
    , m_Value(i_InitialValue)
  {
  }

  //----------------------------------------------------------------------------
  // Get value of property
  //----------------------------------------------------------------------------
  inline const ValueType& GetValue() const { return m_Value; }

  //----------------------------------------------------------------------------
  // Set value of property. The boolean flag is true
  //	if the change is coming from the user interface and therefore
  //	should mark the document containing the property as dirty.
  //----------------------------------------------------------------------------
  // void SetValue(const ValueType& i_Value, UndoFlags i_Undoable = eNoUndo)
  void SetValue(const ValueType& i_Value, bool i_bDirty = false)
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
  void SetValueWithoutNotify(const ValueType& i_Value) { m_Value = i_Value; }

  //--------------------------------------------------------------------
  // Create and return undo operation of correct type for this property.
  // Ownership passes to the caller.
  //--------------------------------------------------------------------
  undoUndoOperation* CreateUndoOperation(std::shared_ptr<prtyPropertyReference> i_pPropertyRef)
  {
    return new prtyUndoTemplate<prtyPropertyTemplate, ValueType>(i_pPropertyRef, this->GetValue());
  }

  prtyPropertyTemplate<ValueType>& operator=(const prtyPropertyTemplate<ValueType>& i_Property)
  {
    // copy base data
    prtyProperty::operator=(i_Property);

    SetValue(i_Property.GetValue());
    return *this;
  }

  prtyPropertyTemplate<ValueType>& operator=(const ValueType& i_Value)
  {
    SetValue(i_Value);
    return *this;
  }

  bool operator==(const prtyPropertyTemplate<ValueType>& i_Property) const
  {
    return GetValue() == i_Property.GetValue();
  }

  bool operator!=(const prtyPropertyTemplate<ValueType>& i_Property) const
  {
    return GetValue() != i_Property.GetValue();
  }

  bool operator==(const ValueType& i_Value) const { return GetValue() == i_Value; }

  bool operator!=(const ValueType& i_Value) const { return GetValue() != i_Value; }

protected:
  ValueType m_Value;
};
