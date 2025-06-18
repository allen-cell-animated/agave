/****************************************************************************\
**	undoUndoInterest.hpp
**
**		An Undo Interest is something that cares about the undo system
**	changing.
**
**
**
\****************************************************************************/
#ifdef UNDO_UNDOINTEREST_HPP
#error undoUndoInterest.hpp multiply included
#endif
#define UNDO_UNDOINTEREST_HPP

//============================================================================
//============================================================================
class undoUndoInterest
{
public:
  //--------------------------------------------------------------------
  //	UndoAdded - undo event added to the stack
  //--------------------------------------------------------------------
  virtual void UndoAdded() = 0;

  //--------------------------------------------------------------------
  //	StackChanged - undo stack changed
  //--------------------------------------------------------------------
  virtual void StackChanged() = 0;
};
