#include "core/prty/prtyRotationUndo.hpp"

#include "core/prty/prtyRotation.hpp"

//--------------------------------------------------------------------
// constructor takes old to be restored if undone
//--------------------------------------------------------------------
prtyRotationUndo::prtyRotationUndo(std::shared_ptr<prtyPropertyReference> i_pPropertyRef,
                                   const glm::vec3& i_EulerBackup)
  : m_pPropertyRef(i_pPropertyRef)
  , m_EulerBackup(i_EulerBackup)
{
}

//--------------------------------------------------------------------
//  Get Name for the operation
//--------------------------------------------------------------------
std::string
prtyRotationUndo::GetDisplayName()
{
  prtyProperty* pProperty = (m_pPropertyRef) ? m_pPropertyRef->GetProperty() : NULL;
  return ((pProperty) ? pProperty->GetPropertyName() : "");
}

//--------------------------------------------------------------------
//  Get memory usage for this operation (in KB). This can be
// accurate or approximate.
//--------------------------------------------------------------------
float
prtyRotationUndo::GetMemoryUsage()
{
  return (undoUndoOperation::GetMemoryUsage() + sizeof(glm::vec3));
  // return (sizeof(bool) / 1000.0f); // convert to KB
}

//--------------------------------------------------------------------
//  Undo is called on an operation when the user chooses
// Edit->Undo from the menu.
//--------------------------------------------------------------------
void
prtyRotationUndo::Undo()
{
  prtyProperty* pPropertyBase = (m_pPropertyRef) ? m_pPropertyRef->GetProperty() : NULL;
  if (pPropertyBase) {
    prtyRotation* pProperty = dynamic_cast<prtyRotation*>(pPropertyBase);
    if (pProperty) {
      float x = 0, y = 0, z = 0;
      pProperty->GetEuler(x, y, z);

      const bool bFromUndo = true;
      pProperty->SetEuler(m_EulerBackup.x, m_EulerBackup.y, m_EulerBackup.z, bFromUndo);

      m_EulerBackup = glm::vec3(x, y, z); // switch backup from undo to redo
    }
  }
}

//--------------------------------------------------------------------
// Redo is called on an operation when the user chooses
// Edit->Redo from the menu and this operation is the next in
// line to be redone.
//--------------------------------------------------------------------
void
prtyRotationUndo::Redo()
{
  prtyProperty* pPropertyBase = (m_pPropertyRef) ? m_pPropertyRef->GetProperty() : NULL;
  if (pPropertyBase) {
    prtyRotation* pProperty = dynamic_cast<prtyRotation*>(pPropertyBase);
    if (pProperty) {
      float x = 0, y = 0, z = 0;
      pProperty->GetEuler(x, y, z);

      const bool bFromUndo = true;
      pProperty->SetEuler(m_EulerBackup.x, m_EulerBackup.y, m_EulerBackup.z, bFromUndo);

      m_EulerBackup = glm::vec3(x, y, z); // switch backup from undo to redo
    }
  }
}

//--------------------------------------------------------------------
//  Commit is called on an operation when it is no longer
// possible for the user to undo this operation.  The
// destructor will soon be called.
//--------------------------------------------------------------------
void
prtyRotationUndo::Commit()
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
void
prtyRotationUndo::Destroy()
{
  // nothing needed
}
