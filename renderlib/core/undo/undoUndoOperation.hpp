/*****************************************************************************
**	undoUndoOperation.hpp
**
**	 This is the abstract base class for operations that can
**	be undone
**
**
**
\****************************************************************************/
#ifdef UNDO_UNDOOPERATION_HPP
#error undoUndoOperation.hpp multiply included
#endif
#define UNDO_UNDOOPERATION_HPP

#include <string>

//============================================================================
//============================================================================
class undoUndoOperation
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  undoUndoOperation();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual ~undoUndoOperation();

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
  virtual void Undo() = 0;

  //--------------------------------------------------------------------
  // Redo is called on an operation when the user chooses
  // Edit->Redo from the menu and this operation is the next in
  // line to be redone.
  //--------------------------------------------------------------------
  virtual void Redo() = 0;

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
};
