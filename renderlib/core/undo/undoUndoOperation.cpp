#include "core/undo/undoUndoOperation.hpp"

//--------------------------------------------------------------------
//--------------------------------------------------------------------
undoUndoOperation::undoUndoOperation() {}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
undoUndoOperation::~undoUndoOperation() {}

//--------------------------------------------------------------------
// Get Name for the operation
//--------------------------------------------------------------------
std::string
undoUndoOperation::GetDisplayName()
{
  return "";
}

//--------------------------------------------------------------------
// Get memory usage for this operation (in KB). This can be accurate
// or approximate.
//--------------------------------------------------------------------
float
undoUndoOperation::GetMemoryUsage()
{
  return 0.0f;
}

//--------------------------------------------------------------------
//  Commit is called on an operation when it is no longer
// possible for the user to undo this operation.  The
// destructor will soon be called. This may happen if the
// history gets too long, or the file is saved.
//--------------------------------------------------------------------
void
undoUndoOperation::Commit()
{
}

//--------------------------------------------------------------------
//  Destroy is called on an operation when it has been undone
// and it can no longer be redone. This may happen after
// a new operation is made when its current state is "undone".
// The destructor will soon be called.
//--------------------------------------------------------------------
void
undoUndoOperation::Destroy()
{
}