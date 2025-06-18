/*****************************************************************************
**	prtyUndoTemplate.hpp
**
**		store a state of Text for undo/redo
**
**
**
\****************************************************************************/
#pragma once
#ifdef PRTY_UNDOTEMPLATE_HPP
#error prtyUndoTemplate.hpp multiply included
#endif
#define PRTY_UNDOTEMPLATE_HPP

#ifndef UNDO_UNDOOPERATION_HPP
#include "core/undo/undoUndoOperation.hpp"
#endif
#ifndef PRTY_PROPERTYREFERENCE_HPP
#include "core/prty/prtyPropertyReference.hpp"
#endif

#include "core/prty/prtyProperty.hpp"

//============================================================================
//============================================================================
template<typename PropertyType, typename BackupType>
class prtyUndoTemplate : public undoUndoOperation
{
public:
  //--------------------------------------------------------------------
  // constructor takes old to be restored if undone
  //--------------------------------------------------------------------
  prtyUndoTemplate(std::shared_ptr<prtyPropertyReference> i_pPropertyRef, const BackupType& i_PropertyBackup)
    : m_pPropertyRef(i_pPropertyRef)
    , m_PropertyBackup(i_PropertyBackup)
  {
  }

  //--------------------------------------------------------------------
  //  Get Name for the operation
  //--------------------------------------------------------------------
  std::string GetDisplayName()
  {
    prtyProperty* pProperty = (m_pPropertyRef) ? m_pPropertyRef->GetProperty() : NULL;
    return ((pProperty) ? pProperty->GetPropertyName() : "");
  }

  //--------------------------------------------------------------------
  //  Get memory usage for this operation (in KB). This can be
  // accurate or approximate.
  //--------------------------------------------------------------------
  float GetMemoryUsage()
  {
    return (sizeof(BackupType) / 1000.0f); // convert to KB
  }

  //--------------------------------------------------------------------
  //  Undo is called on an operation when the user chooses
  // Edit->Undo from the menu.
  //--------------------------------------------------------------------
  void Undo()
  {
    PropertyType* pProperty = resolve_property();
    if (pProperty) {
      BackupType temp = pProperty->GetValue();
      const bool bFromUndo = true;
      pProperty->SetValue(m_PropertyBackup, bFromUndo);
      m_PropertyBackup = temp; // switch backup from undo to redo
    }
  }

  //--------------------------------------------------------------------
  // Redo is called on an operation when the user chooses
  // Edit->Redo from the menu and this operation is the next in
  // line to be redone.
  //--------------------------------------------------------------------
  void Redo()
  {
    PropertyType* pProperty = resolve_property();
    if (pProperty) {
      BackupType temp = pProperty->GetValue();
      const bool bFromUndo = true;
      pProperty->SetValue(m_PropertyBackup, bFromUndo);
      m_PropertyBackup = temp; // switch backup from undo to redo
    }
  }

  //--------------------------------------------------------------------
  //  Commit is called on an operation when it is no longer
  // possible for the user to undo this operation.  The
  // destructor will soon be called.
  //--------------------------------------------------------------------
  void Commit()
  {
    // nothing needed
  }

  //--------------------------------------------------------------------
  //  Destroy is called on an operation when it has been undone
  // and it can no longer be redone. This may happen after the
  // history gets long enough or a new operation is made when
  // its current state is "undone". The destructor will soon be
  // called.
  //--------------------------------------------------------------------
  void Destroy()
  {
    // nothing needed
  }

private:
  //--------------------------------------------------------------------
  // convert property reference into a property of our type
  //--------------------------------------------------------------------
  PropertyType* resolve_property()
  {
    if (m_pPropertyRef) {
      prtyProperty* pProperty = m_pPropertyRef->GetProperty();
      if (pProperty)
        return dynamic_cast<PropertyType*>(pProperty);
    }
    return NULL;
  }

  std::shared_ptr<prtyPropertyReference> m_pPropertyRef;
  BackupType m_PropertyBackup;
};
