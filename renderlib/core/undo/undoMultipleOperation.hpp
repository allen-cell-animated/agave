/*****************************************************************************
**	undoMultipleOperation.hpp
**
**	 This class can be used to group a number of single undo operations
**	into one undo operation that can be undone and redone in one step.
**
**
**
\****************************************************************************/
#ifdef UNDO_MULTIPLEOPERATION_HPP
#error undoUndoOperation.hpp multiply included
#endif
#define UNDO_MULTIPLEOPERATION_HPP

#ifndef UNDO_UNDOOPERATION_HPP
#include "core/undo/undoUndoOperation.hpp"
#endif

#include <vector>

//============================================================================
//============================================================================
class undoMultipleOperation : public undoUndoOperation
{
public:
  //--------------------------------------------------------------------
  // If a name is passed in here, then it will be used for the
  // operation block as a whole. Otherwise, the name will come from
  // the first operation's display name.
  //--------------------------------------------------------------------
  undoMultipleOperation(const char* i_Name = NULL);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual ~undoMultipleOperation();

  //--------------------------------------------------------------------
  // Add operation to this multiple operation block.
  // Ownership passes to this object.
  //--------------------------------------------------------------------
  void AddOperation(undoUndoOperation* i_pOperation);

  //--------------------------------------------------------------------
  // Returns number of operations in the stack.
  //--------------------------------------------------------------------
  int GetNumOperations();

  //--------------------------------------------------------------------
  // Returns true iff the operation is in the list of operations
  //	in this block.
  //--------------------------------------------------------------------
  bool ContainsOperation(undoUndoOperation* i_pOperation);

  //--------------------------------------------------------------------
  // Get Name for the operation
  //--------------------------------------------------------------------
  virtual std::string GetDisplayName();

  //--------------------------------------------------------------------
  // Get memory usage for this operation (in KB). This can be accurate
  // or approximate.
  //--------------------------------------------------------------------
  virtual float GetMemoryUsage();

  //--------------------------------------------------------------------
  // Undo is called on an operation when the user chooses
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
  // destructor will soon be called. This may happen if the
  // history gets too long, or the file is saved.
  //--------------------------------------------------------------------
  virtual void Commit();

  //--------------------------------------------------------------------
  //  Destroy is called on an operation when it has been undone
  // and it can no longer be redone. This may happen after
  // a new operation is made when its current state is "undone".
  // The destructor will soon be called.
  //--------------------------------------------------------------------
  virtual void Destroy();

private:
  std::string m_Name;
  std::vector<undoUndoOperation*> m_Operations;
};
