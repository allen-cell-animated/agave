#pragma once

#include "core/prty/prtyPropertyCallback.hpp"
#include "core/undo/undoUndoOperation.hpp"

#include <list>
#include <memory>
#include <string>
#include <vector>

//============================================================================
//	forward references
//============================================================================
class prtyProperty;
class prtyPropertyReference;
class docReader;
class docWriter;

//============================================================================
//	typedefs
//============================================================================
namespace {
typedef std::vector<prtyProperty*> property_list;
};

//============================================================================
/****************************************************************************\
**
**		A property corresponds to a value, singular or compound.
**	(int, float, vector, matrix, color, etc.)
**
**	How to create a new property:
**	1) Create new property class
**		a) a child of prtyProperty
**		b) 2 constructors: one takes the name, other takes name + initial value
**		c) GetValue and SetValue. SetValue should handle undo or not
**		d) operators (at least =) [other useful ==, !=]
**		e) the data itself
**	2) Create the property's undo class
**	3) [optional] Create at least one control connection for the property
**		a) a class that ties the property to a specific control
**		b) implement this classes instantiation in a control factory
**
**	How to create a new control:
**	1) Create new property-control class
**		a) a child of prtyControl
**	2) Create the instantiation of the control in appropriate Control Factory
**		a) make sure at correct property type is checked (from #1)
**	3) Create a UIInfo class for this control
**
**	How to USE a new property
**	1) add the property to whatever data class is going to hold it (like a struct
**		of properties) [Also update any parser files]
**	2) some class must have a parent of prtyObject and then register the
**		properties *only if* they are going to be used as controls that
**		will be built dynamically.
**		a) if a property is going to be displayed on a Form then it must be added
**			to the control list.
**	3) to build the form make the call:
**		prtyFormControlBuilder::BuildForm( pControl, (m_pObject->GetListContainer()) );
**			where pControl is a System::Windows::Forms::Control*
**			and m_pObject->GetListContainer() is the container of registered properties
**
**	Applications using prty must:
**	1) call these functions to initialize
**		prtyCallbackMgr::Initialize();
**		prtyControlMgr::Initialize();
**		prtyControlMgr::AddControlFactory( new prtyControlFactoryBase() );
**		prtyControlMgr::AddControlFactory( new prtyControlFactoryTMC() );
**	2) 	call these functions to deinitialize
**		prtyControlMgr::DeInitialize();
**		prtyCallbackMgr::DeInitialize();
**
**	Refer to test/prty application for an example.
**
**
**
\****************************************************************************/
//============================================================================
class prtyProperty
{
public:
  // Undo is not created on a property anymore, use the prtyObject to
  // create undo operations. SetValue() functions should just take a boolean
  // for whether the change is coming from the user interface in order to
  // set a dirty bit for a document.
  //
  // enum UndoFlags
  //{
  //	eNoUndo = 0,
  //	eContinueUndo,		// continue current undo operation
  //	eNewUndo,			// new operation
  //	eFromUndo			// change is coming from an undo operation being undone or redone
  //};

public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyProperty();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyProperty(const std::string& i_Name);

  //--------------------------------------------------------------------
  // copy constructor does not copy callbacks
  //--------------------------------------------------------------------
  prtyProperty(const prtyProperty& i_Property);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  ~prtyProperty();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  const std::string& GetPropertyName() const;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  void SetPropertyName(const std::string& i_Name);

  //--------------------------------------------------------------------
  // Animatable - can this property be animated?
  //--------------------------------------------------------------------
  bool IsAnimatable() const;
  void SetAnimatable(bool i_bVal);

  //--------------------------------------------------------------------
  // Scripted - Is this property currently being animated?
  //--------------------------------------------------------------------
  bool IsScripted() const;
  void SetScripted(bool i_bVal);

  //--------------------------------------------------------------------
  // ValueVariation - Does the current value vary from the scripted value?
  //--------------------------------------------------------------------
  bool HasValueVariation() const;
  void SetValueVariation(bool i_bVal);

  //--------------------------------------------------------------------
  //	assignment operator - does not copy the callbacks
  //--------------------------------------------------------------------
  prtyProperty& operator=(const prtyProperty& i_Property);

  //--------------------------------------------------------------------
  // AddCallback using interface.
  //
  //	NOTE: This property will take ownership of the callback
  //	pointer and will delete it in its destructor! It should be
  //	created on the heap and probably should use the
  //	prtyCallbackWrapper template.
  //--------------------------------------------------------------------
  void AddCallback(prtyPropertyCallback* i_pCallback);

  //--------------------------------------------------------------------
  // AddCallback shared pointer variation for when you want to
  //	keep a copy of the callback also.
  //--------------------------------------------------------------------
  void AddCallback(std::shared_ptr<prtyPropertyCallback> i_pCallback);

  //--------------------------------------------------------------------
  // RemoveCallback by shared_ptr. You should be keeping a shared_ptr
  //	to the callback passed to AddCallback() if you want to remove
  //	it later. Don't use the direct pointer function when adding
  //	if you want to keep a copy.
  //--------------------------------------------------------------------
  void RemoveCallback(std::shared_ptr<prtyPropertyCallback> i_pCallback);

  //--------------------------------------------------------------------
  //	Call all the callbacks -
  //		i_bWithUndo should be set to (i_Undoable != eNoUndo)
  //	in derived classes SetValue() functions.
  //--------------------------------------------------------------------
  void NotifyCallbacksPropertyChanged(bool i_bWithUndo);

  //--------------------------------------------------------------------
  //	The type of property it is
  //--------------------------------------------------------------------
  virtual const char* GetType() = 0;

  //--------------------------------------------------------------------
  // Create an undo operation of the correct type for this
  // property. A reference to this property should be passed in.
  // Ownership passes to the caller.
  //--------------------------------------------------------------------
  virtual undoUndoOperation* CreateUndoOperation(std::shared_ptr<prtyPropertyReference> i_pPropertyRef) = 0;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Read(docReader& io_Reader) = 0;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Write(docWriter& io_Writer) const = 0;

  //--------------------------------------------------------------------
  //  In some cases we want the form builder to check whether or not
  // this property should be visible.
  //--------------------------------------------------------------------
  void SetVisible(bool i_bVisible);
  bool GetVisible() const;

protected:
  //--------------------------------------------------------------------
  //	The type of property it is
  //--------------------------------------------------------------------
  // void SetType(const std::string i_Type);

protected:
  /*static*/ undoUndoOperation* ms_LastUndoOp;

private:
  // std::string	m_Type;				// type of property (each property class)
  std::string m_Name; // name (situation-specific) of property
  std::vector<std::shared_ptr<prtyPropertyCallback>> m_CallbackObjects;
  bool m_bVisible;
  // Animation state flags
  struct
  {
    bool m_bAnimatable : 1;     // can this property be animated?
    bool m_bScripted : 1;       // Is this property currently being animated?
    bool m_bValueVariation : 1; // Does the current value vary from the scripted value?
  } m_Flags;
};
