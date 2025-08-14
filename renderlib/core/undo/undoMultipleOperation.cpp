#include "core/undo/undoMultipleOperation.hpp"

#include "Logging.h"
#include "core/env/envSTLHelpers.hpp"

//--------------------------------------------------------------------
// If a name is passed in here, then it will be used for the
// operation block as a whole. Otherwise, the name will come from
// the first operation's display name.
//--------------------------------------------------------------------
undoMultipleOperation::undoMultipleOperation(const char* i_Name)
{
  if (i_Name != NULL)
    m_Name = i_Name;
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
undoMultipleOperation::~undoMultipleOperation()
{
  envSTLHelpers::DeleteContainer(m_Operations);
}

//--------------------------------------------------------------------
// Add operation to this multiple operation block.
// Ownership passes to this object.
//--------------------------------------------------------------------
void
undoMultipleOperation::AddOperation(undoUndoOperation* i_pOperation)
{
  DBG_ASSERT(i_pOperation, "NULL operation passed to AddOperation");
  m_Operations.push_back(i_pOperation);

  //	DBG_LOG("AddOperation, new size: " << m_Operations.size());
}

//--------------------------------------------------------------------
// Returns number of operations in the stack.
//--------------------------------------------------------------------
int
undoMultipleOperation::GetNumOperations()
{
  return m_Operations.size();
}

//--------------------------------------------------------------------
// Returns true iff the operation is in the list of operations
//	in this block.
//--------------------------------------------------------------------
bool
undoMultipleOperation::ContainsOperation(undoUndoOperation* i_pOperation)
{
  return envSTLHelpers::Contains(m_Operations, i_pOperation);
}

//--------------------------------------------------------------------
// Get Name for the operation
//--------------------------------------------------------------------
std::string
undoMultipleOperation::GetDisplayName()
{
  //	DBG_LOG("GetDisplayName, size: " << m_Operations.size());

  if (m_Name.empty() && !m_Operations.empty()) {
    return m_Operations[0]->GetDisplayName();
  }
  return m_Name;
}

//--------------------------------------------------------------------
// Get memory usage for this operation (in KB). This can be accurate
// or approximate.
//--------------------------------------------------------------------
float
undoMultipleOperation::GetMemoryUsage()
{
  float sum = 0.0f;
  const int num_ops = m_Operations.size();
  for (int i = 0; i < num_ops; i++) {
    sum += m_Operations[i]->GetMemoryUsage();
  }
  return sum;
}

//--------------------------------------------------------------------
// Undo is called on an operation when the user chooses
// Edit->Undo from the menu.
//--------------------------------------------------------------------
void
undoMultipleOperation::Undo()
{
  // bga- undo has to be done in reverse in order to restore original state correctly
  // envSTLHelpers::ForAll(m_Operations, envSTLHelpers::MemFun(&undoUndoOperation::Undo));
  const int num_ops = m_Operations.size();
  for (int i = num_ops - 1; i >= 0; i--) {
    m_Operations[i]->Undo();
  }
}

//--------------------------------------------------------------------
// Redo is called on an operation when the user chooses
// Edit->Redo from the menu and this operation is the next in
// line to be redone.
//--------------------------------------------------------------------
void
undoMultipleOperation::Redo()
{
  envSTLHelpers::ForAll(m_Operations, std::mem_fn(&undoUndoOperation::Redo));
}

//--------------------------------------------------------------------
//  Commit is called on an operation when it is no longer
// possible for the user to undo this operation.  The
// destructor will soon be called. This may happen if the
// history gets too long, or the file is saved.
//--------------------------------------------------------------------
void
undoMultipleOperation::Commit()
{
  envSTLHelpers::ForAll(m_Operations, std::mem_fn(&undoUndoOperation::Commit));
}

//--------------------------------------------------------------------
//  Destroy is called on an operation when it has been undone
// and it can no longer be redone. This may happen after
// a new operation is made when its current state is "undone".
// The destructor will soon be called.
//--------------------------------------------------------------------
void
undoMultipleOperation::Destroy()
{
  envSTLHelpers::ForAll(m_Operations, std::mem_fn(&undoUndoOperation::Destroy));
}