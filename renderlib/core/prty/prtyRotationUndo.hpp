/*****************************************************************************
**	prtyRotationUndo.hpp
**
**		store a state of Rotation for undo/redo
**
**
**
\****************************************************************************/
#pragma once
#ifdef PRTY_ROTATIONUNDO_HPP
#error prtyRotationUndo.hpp multiply included
#endif
#define PRTY_ROTATIONUNDO_HPP

#ifndef UNDO_UNDOOPERATION_HPP
#include "core/undo/undoUndoOperation.hpp"
#endif
#ifndef PRTY_PROPERTYREFERENCE_HPP
#include "core/prty/prtyPropertyReference.hpp"
#endif

#include "glm.h"

//============================================================================
//============================================================================
class prtyRotationUndo : public undoUndoOperation
{
public:
  //--------------------------------------------------------------------
  // constructor takes old to be restored if undone
  //--------------------------------------------------------------------
  prtyRotationUndo(std::shared_ptr<prtyPropertyReference> i_pPropertyRef, const glm::vec3& i_EulerBackup);

  //--------------------------------------------------------------------
  //  Get Name for the operation
  //--------------------------------------------------------------------
  virtual std::string GetDisplayName();

  //--------------------------------------------------------------------
  //  Get memory usage for this operation (in KB). This can be
  // accurate or approximate.
  //--------------------------------------------------------------------
  virtual float GetMemoryUsage();

  //--------------------------------------------------------------------
  //  Undo is called on an operation when the user chooses
  // Edit->Undo from the menu.
  //--------------------------------------------------------------------
  virtual void Undo();

  //--------------------------------------------------------------------
  // Redo is called on an operation when the user chooses
  // Edit->Redo from the menu and this operation is the next in
  // line to be redone.
  //--------------------------------------------------------------------
  virtual void Redo();

  //--------------------------------------------------------------------
  //  Commit is called on an operation when it is no longer
  // possible for the user to undo this operation.  The
  // destructor will soon be called.
  //--------------------------------------------------------------------
  virtual void Commit();

  //--------------------------------------------------------------------
  //  Destroy is called on an operation when it has been undone
  // and it can no longer be redone. This may happen after the
  // history gets long enough or a new operation is made when
  // its current state is "undone". The destructor will soon be
  // called.
  //--------------------------------------------------------------------
  virtual void Destroy();

private:
  std::shared_ptr<prtyPropertyReference> m_pPropertyRef;
  glm::vec3 m_EulerBackup;
};
