/*****************************************************************************
**	undoUndoMgr.cpp
**
**		see .hpp
**
**
**
\****************************************************************************/
#include "core/undo/undoUndoMgr.hpp"

#include "core/undo/undoMultipleOperation.hpp"
#include "core/undo/undoUndoInterest.hpp"

#include "Logging.h"
#include "core/env/envSTLHelpers.hpp"

#include <deque>
#include <stack>

//============================================================================
//============================================================================
namespace undoUndoMgr {
namespace {
// Use deque to put new operations in front, removing
// off of the back to reduce memory and tracking the
// number of undone operation with a counter that
// can be indexed into deque from the front.
std::deque<undoUndoOperation*> l_Operations;
int l_NumUndos = 0;

// Stack of open multiple operation blocks
std::stack<undoMultipleOperation*> l_BlockUndos;

// limits on size of deque
float l_MemoryMaxUsage = 0;
int l_MinSizeOfUndoStack = 40;

float l_CurrentMemoryUsage = 0;

std::vector<undoUndoInterest*> l_Interests;

//--------------------------------------------------------------------
//--------------------------------------------------------------------
void
notify_interests_undo_added()
{
  int num = l_Interests.size();
  for (int i = 0; i < num; i++) {
    l_Interests[i]->UndoAdded();
  }
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
void
notify_interests_stack_changed()
{
  int num = l_Interests.size();
  for (int i = 0; i < num; i++) {
    l_Interests[i]->StackChanged();
  }
}

//--------------------------------------------------------------------
// Call Destroy on undone operations, delete them
// and remove them from the list
//--------------------------------------------------------------------
void
destroy_undone()
{
  for (int i = 0; i < l_NumUndos; i++) {
    undoUndoOperation* op = l_Operations.front();
    l_CurrentMemoryUsage -= op->GetMemoryUsage();
    op->Destroy();
    delete op;
    l_Operations.pop_front();
  }
  l_NumUndos = 0;

  notify_interests_stack_changed();
}

//--------------------------------------------------------------------
// add operation to stack
//--------------------------------------------------------------------
void
add_operation(undoUndoOperation* i_Op)
{
  if (l_NumUndos > 0) {
    // Destroy undone operations
    destroy_undone();
  }

  // Track memory usage and maybe remove operations from
  // the back, calling Commit() on them
  while ((l_CurrentMemoryUsage > l_MemoryMaxUsage) && (l_Operations.size() > l_MinSizeOfUndoStack)) {
    undoUndoOperation* op = l_Operations.back();
    l_CurrentMemoryUsage -= op->GetMemoryUsage();
    op->Commit();
    delete op;
    l_Operations.pop_back();
  }

  l_Operations.push_front(i_Op);
  l_CurrentMemoryUsage += i_Op->GetMemoryUsage();
}
} // end of namespace

//--------------------------------------------------------------------
// Set memory limit for history buffer (in KB) except it is allowed
// to go over the limit in order to maintain at least a certain
// minimum size to the undo stack.
//--------------------------------------------------------------------
void
SetLimits(float i_MemoryMaxUsage, int i_MinSizeOfUndoStack)
{
  l_MemoryMaxUsage = i_MemoryMaxUsage;
  l_MinSizeOfUndoStack = i_MinSizeOfUndoStack;
  if (l_MinSizeOfUndoStack < 0)
    l_MinSizeOfUndoStack = 0;
}

//--------------------------------------------------------------------
//  Call AddOperation to add an undo step to the stack
//--------------------------------------------------------------------
void
AddOperation(undoUndoOperation* i_Op)
{
  if (!l_BlockUndos.empty()) {
    undoMultipleOperation* multi_op = l_BlockUndos.top();
    multi_op->AddOperation(i_Op);
  } else {
    add_operation(i_Op);
    notify_interests_undo_added();
  }
}

//--------------------------------------------------------------------
// BeginMultipleOperationBlock - subsequent calls to AddOperation
//	will be added to a undoMultipleOperation block. After the
//	call to EndMultipleOperationBlock, this set of operations
//	will appear as one operation in the stack.
//--------------------------------------------------------------------
void
BeginMultipleOperationBlock(const char* i_Name)
{
  undoMultipleOperation* multi_op = new undoMultipleOperation(i_Name);
  l_BlockUndos.push(multi_op);
}
void
EndMultipleOperationBlock()
{
  undoMultipleOperation* multi_op = l_BlockUndos.top();
  DBG_ASSERT(multi_op, "EndMultipleOperationBlock does not match number of Begin calls");
  l_BlockUndos.pop();

  // Only add the multi-op if stack is finished and
  // some operations have been added to it.
  if (l_BlockUndos.empty() && multi_op->GetNumOperations() > 0) {
    add_operation(multi_op);
    notify_interests_undo_added();
  } else {
    // If we don't add the operation, need to delete it.
    delete multi_op;
  }
}

//--------------------------------------------------------------------
// Returns pointer to last operation to have been added. May return NULL
// if the buffer is cleared, or if some operations have been undone.
//--------------------------------------------------------------------
undoUndoOperation*
PeekLastOperation()
{
  if (l_Operations.empty())
    return NULL;
  if (l_NumUndos > 0)
    return NULL;
  return l_Operations.front();
}

//--------------------------------------------------------------------
// See if the given operation is either the last operation, or is
//	part of a multiple undo block that is the last operation.
//--------------------------------------------------------------------
bool
IsInLastOperation(undoUndoOperation* i_pOperation)
{
  if (!i_pOperation)
    return false;

  undoUndoOperation* pLastOp = PeekLastOperation();
  if (pLastOp == i_pOperation)
    return true;

  // If the last operation is a multiple undo block, then
  // look for the operation in this block.
  undoMultipleOperation* pMultiOp = dynamic_cast<undoMultipleOperation*>(pLastOp);
  if (pMultiOp) {
    bool found = pMultiOp->ContainsOperation(i_pOperation);
    return found;
  }

  return false;
}

//--------------------------------------------------------------------
//  Call Commit when it is no longer possible for the user to
// undo any operation. This should be called when an undoable
// operation takes place or some serializing (read or write)
// has taken place
//--------------------------------------------------------------------
void
Commit()
{
  if (l_NumUndos > 0) {
    // Destroy undone operations
    destroy_undone();
  }

  // Call Commit on all other operations, delete them
  // and clear the whole queue
  while (!l_Operations.empty()) {
    undoUndoOperation* op = l_Operations.back();
    l_CurrentMemoryUsage -= op->GetMemoryUsage();
    op->Commit();
    delete op;
    l_Operations.pop_back();
  }

  notify_interests_stack_changed();
}

//--------------------------------------------------------------------
// Call Undo when the user chooses Edit->Undo from the menu.
//--------------------------------------------------------------------
void
Undo()
{
  if (l_NumUndos < l_Operations.size()) {
    l_Operations[l_NumUndos]->Undo();
    l_NumUndos++;

    notify_interests_stack_changed();
  }
}

//--------------------------------------------------------------------
//  Use this to set the Enabled state of the Edit->Undo button
//--------------------------------------------------------------------
bool
CanUndo()
{
  return (l_Operations.size() > l_NumUndos);
}

//--------------------------------------------------------------------
// Returns display string for the type of operation to be undone
// if Undo were to be called.
//--------------------------------------------------------------------
std::string
GetUndoOperationName()
{
  if (l_NumUndos < l_Operations.size()) {
    return l_Operations[l_NumUndos]->GetDisplayName();
  }
  return "";
}

//--------------------------------------------------------------------
// Returns the index in the stack for the next undo operation.
// -1 is returned if the stack is empty or nothing left to undo.
//--------------------------------------------------------------------
int
GetUndoOperationIndex()
{
  if (l_NumUndos < l_Operations.size()) {
    return l_NumUndos;
  }
  return -1;
}

//--------------------------------------------------------------------
// Call Redo when the user chooses Edit->Redo from the menu
//--------------------------------------------------------------------
void
Redo()
{
  if (l_NumUndos > 0) {
    l_NumUndos--;
    l_Operations[l_NumUndos]->Redo();

    notify_interests_stack_changed();
  }
}

//--------------------------------------------------------------------
//  Use this to set the Enabled state of the Edit->Redo button
//--------------------------------------------------------------------
bool
CanRedo()
{
  return (l_NumUndos > 0);
}

//--------------------------------------------------------------------
// Returns display string for the type of operation to be redone
// if Redo were to be called.
//--------------------------------------------------------------------
std::string
GetRedoOperationName()
{
  if (l_NumUndos > 0) {
    return l_Operations[l_NumUndos - 1]->GetDisplayName();
  }
  return "";
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
void
GetUndoStack(std::vector<std::string>& i_Stack)
{
  i_Stack.resize(l_Operations.size());

  for (int i = 0; i < l_Operations.size(); i++) {
    undoUndoOperation* op = l_Operations[i];
    i_Stack[i] = op->GetDisplayName();
  }
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
float
GetMaximumMemoryUsage()
{
  return l_MemoryMaxUsage;
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
float
GetCurrentMemoryUsage()
{
  return l_CurrentMemoryUsage;
}

//--------------------------------------------------------------------
//	Add/RemoveInterest() - add or remove an interest for Undo
//--------------------------------------------------------------------
void
AddInterest(undoUndoInterest* i_pInterest)
{
  l_Interests.push_back(i_pInterest);
}
void
RemoveInterest(undoUndoInterest* i_pInterest)
{
  envSTLHelpers::RemoveOneValue(l_Interests, i_pInterest);
}

} // end of namespace
