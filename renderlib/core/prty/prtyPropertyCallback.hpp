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
#ifdef PRTY_PROPERTYCALLBACK_HPP
#error prtyPropertyCallback.hpp multiply included
#endif
#define PRTY_PROPERTYCALLBACK_HPP

//============================================================================
//	forward references
//============================================================================
class prtyProperty;

//============================================================================
//	Interface
//============================================================================
class prtyPropertyCallback
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual ~prtyPropertyCallback() = 0;

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
