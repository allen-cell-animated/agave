#pragma once

#include <functional>

//============================================================================
//	forward references
//============================================================================
class prtyProperty;

typedef std::function<void(prtyProperty*, bool)> prtyPropertyCallbackFunc;

//============================================================================
//	Interface
//============================================================================
class prtyPropertyCallback
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual ~prtyPropertyCallback() {}

  //--------------------------------------------------------------------
  // PropertyChanged is called when a property's value is changed
  //	through a "SetValue" function. The boolean flag is true
  //	if the change is coming from the user interface and therefore
  //	should mark the document containing the property as dirty.
  //--------------------------------------------------------------------
  virtual void PropertyChanged(prtyProperty* i_pProperty, bool i_bDirty) = 0;
};

//============================================================================
// Wrapper template
//============================================================================
template<class T>
struct prtyCallbackWrapper : public prtyPropertyCallback
{
  prtyCallbackWrapper(T* i_this, void (T::*i_memberFunc)(prtyProperty*, bool))
    : obj_ptr(i_this)
    , func_ptr(i_memberFunc)
  {
  }

  T* obj_ptr;
  void (T::*func_ptr)(prtyProperty*, bool);

  virtual void PropertyChanged(prtyProperty* i_pProperty, bool i_bWithUndo)
  {
    (obj_ptr->*func_ptr)(i_pProperty, i_bWithUndo);
  }
};

class prtyCallbackLambda : public prtyPropertyCallback
{
  prtyPropertyCallbackFunc m_Callback;

public:
  prtyCallbackLambda(prtyPropertyCallbackFunc i_Callback)
    : m_Callback(i_Callback)
  {
  }

  virtual void PropertyChanged(prtyProperty* i_pProperty, bool i_bWithUndo) override
  {
    m_Callback(i_pProperty, i_bWithUndo);
  }
};