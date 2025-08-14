#pragma once

#include <string>
#include <vector>

//============================================================================
//============================================================================
class undoUndoInterest;
class undoUndoOperation;

//============================================================================
/*****************************************************************************
**
**	 An UndoOperation should be created each time a step is
**	taken that can be undone. The UndoMgr keeps track of the
**	last few operations.
**
\****************************************************************************/
//============================================================================
namespace undoUndoMgr {
//--------------------------------------------------------------------
// Set memory limit for history buffer (in KB) except it is allowed
// to go over the limit in order to maintain at least a certain
// minimum size to the undo stack.
//--------------------------------------------------------------------
void
SetLimits(float i_MemoryMaxUsage, int i_MinSizeOfUndoStack = 3);

//--------------------------------------------------------------------
//  Call AddOperation to add an undo step to the stack
//--------------------------------------------------------------------
void
AddOperation(undoUndoOperation* i_Op);

//--------------------------------------------------------------------
// BeginMultipleOperationBlock - subsequent calls to AddOperation
//	will be added to a undoMultipleOperation block. After the
//	call to EndMultipleOperationBlock, this set of operations
//	will appear as one operation in the stack.
//  Name is optional.
//--------------------------------------------------------------------
void
BeginMultipleOperationBlock(const char* i_Name = NULL);
void
EndMultipleOperationBlock();

//--------------------------------------------------------------------
// Returns pointer to last operation to have been added. May return NULL
// if the buffer is cleared, or if some operations have been undone.
//--------------------------------------------------------------------
undoUndoOperation*
PeekLastOperation();

//--------------------------------------------------------------------
// See if the given operation is either the last operation, or is
//	part of a multiple undo block that is the last operation.
//--------------------------------------------------------------------
bool
IsInLastOperation(undoUndoOperation* i_pOperation);

//--------------------------------------------------------------------
//  Call Commit when it is no longer possible for the user to
// undo any operation. This should be called when an undoable
// operation takes place or some serializing (read or write)
// has taken place
//--------------------------------------------------------------------
void
Commit();

//--------------------------------------------------------------------
// Call Undo when the user chooses Edit->Undo from the menu.
//--------------------------------------------------------------------
void
Undo();

//--------------------------------------------------------------------
//  Use this to set the Enabled state of the Edit->Undo button
//--------------------------------------------------------------------
bool
CanUndo();

//--------------------------------------------------------------------
// Returns display string for the type of operation to be undone
// if Undo were to be called.
//--------------------------------------------------------------------
std::string
GetUndoOperationName();

//--------------------------------------------------------------------
// Returns the index in the stack for the next undo operation.
// -1 is returned if the stack is empty or nothing left to undo.
//--------------------------------------------------------------------
int
GetUndoOperationIndex();

//--------------------------------------------------------------------
// Call Redo when the user chooses Edit->Redo from the menu
//--------------------------------------------------------------------
void
Redo();

//--------------------------------------------------------------------
//  Use this to set the Enabled state of the Edit->Redo button
//--------------------------------------------------------------------
bool
CanRedo();

//--------------------------------------------------------------------
// Returns display string for the type of operation to be redone
// if Redo were to be called.
//--------------------------------------------------------------------
std::string
GetRedoOperationName();

//--------------------------------------------------------------------
//--------------------------------------------------------------------
void
GetUndoStack(std::vector<std::string>& i_Stack);

//--------------------------------------------------------------------
//--------------------------------------------------------------------
float
GetMaximumMemoryUsage();

//--------------------------------------------------------------------
//--------------------------------------------------------------------
float
GetCurrentMemoryUsage();

//--------------------------------------------------------------------
//	Add/RemoveInterest() - add or remove an interest for Undo
//--------------------------------------------------------------------
void
AddInterest(undoUndoInterest* i_pInterest);
void
RemoveInterest(undoUndoInterest* i_pInterest);

} // end of namespace
