/****************************************************************************\
**	prtyPropertyCallback.hpp
**
**	Template and interface for handling property changed callbacks.
**
**	You create and register a callback to a member function with this syntax:
**		m_Property.AddCallback(new prtyCallbackWrapper<xxxObject>(this, &xxxObject::FunctionName));
**
**
**
\****************************************************************************/
#pragma once
#include <functional>

//============================================================================
//	forward references
//============================================================================
class agBaseProperty;

//============================================================================
//	Interface
//============================================================================
class agPropertyCallback
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual ~agPropertyCallback() {}

  //--------------------------------------------------------------------
  // PropertyChanged is called when a property's value is changed
  //	through a "SetValue" function. The boolean flag is true
  //	if the change is coming from the user interface and therefore
  //	should mark the document containing the property as dirty.
  //--------------------------------------------------------------------
  virtual void PropertyChanged(agBaseProperty* i_pProperty, bool i_bDirty) = 0;
};

//============================================================================
// Wrapper template
//============================================================================
template<class T>
struct agPropertyCallbackWrapper : public agPropertyCallback
{
  agPropertyCallbackWrapper(T* i_this, void (T::*i_memberFunc)(agBaseProperty*, bool))
    : obj_ptr(i_this)
    , func_ptr(i_memberFunc)
  {
  }

  T* obj_ptr;
  void (T::*func_ptr)(agBaseProperty*, bool);

  virtual void PropertyChanged(agBaseProperty* i_pProperty, bool i_bWithUndo)
  {
    (obj_ptr->*func_ptr)(i_pProperty, i_bWithUndo);
  }
};

class agPropertyCallbackLambda : public agPropertyCallback
{
  std::function<void(agBaseProperty*, bool)> m_Callback;

public:
  agPropertyCallbackLambda(std::function<void(agBaseProperty*, bool)> i_Callback)
    : m_Callback(i_Callback)
  {
  }

  virtual void PropertyChanged(agBaseProperty* i_pProperty, bool i_bWithUndo) override
  {
    m_Callback(i_pProperty, i_bWithUndo);
  }
};