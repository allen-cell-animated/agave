#pragma once

#include "core/undo/undoUndoMgr.hpp"

//============================================================================
/*****************************************************************************
**
**	 A helper class for begining and ending a multiple operation
**	undo block. The class begin the block on constructor and ends
**	it on destructor so that exceptions don't throw off the block stack.
**
\****************************************************************************/
//============================================================================
class undoMultipleOperationBlock
{
public:
  //--------------------------------------------------------------------
  // Creates a multiple operation undo block if i_bDoBlock is true.
  //--------------------------------------------------------------------
  undoMultipleOperationBlock(const char* i_Name = NULL, bool i_bDoBlock = true)
    : m_bDoBlock(i_bDoBlock)
  {
    if (m_bDoBlock)
      undoUndoMgr::BeginMultipleOperationBlock(i_Name);
  }

  //--------------------------------------------------------------------
  // Ends multiple operation block in destructor if one was started
  // in constructor
  //--------------------------------------------------------------------
  ~undoMultipleOperationBlock()
  {
    if (m_bDoBlock)
      undoUndoMgr::EndMultipleOperationBlock();
  }

private:
  bool m_bDoBlock;
};
