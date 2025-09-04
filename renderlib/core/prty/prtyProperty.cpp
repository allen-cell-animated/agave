#include "core/prty/prtyProperty.hpp"

// #include "core/dbg/dbgMsg.hpp"
#include "core/env/envSTLHelpers.hpp"

//--------------------------------------------------------------------
//--------------------------------------------------------------------
prtyProperty::prtyProperty()
  : m_Name("property")
  , m_bVisible(true)
{
  m_Flags.m_bAnimatable = false;
  m_Flags.m_bScripted = false;
  m_Flags.m_bValueVariation = false;
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
prtyProperty::prtyProperty(const std::string& i_Name)
  : m_Name(i_Name)
  , m_bVisible(true)
{
  m_Flags.m_bAnimatable = false;
  m_Flags.m_bScripted = false;
  m_Flags.m_bValueVariation = false;
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
prtyProperty::prtyProperty(const prtyProperty& i_Property)
  : m_Name(i_Property.m_Name)
  , m_bVisible(i_Property.m_bVisible)
// m_Type(i_Property.m_Type)
{
  m_Flags.m_bAnimatable = false;
  m_Flags.m_bScripted = false;
  m_Flags.m_bValueVariation = false;
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
prtyProperty::~prtyProperty()
{
  // if (!m_CallbackObjects.empty())
  //{
  //	DBG_LOG("Property destructor, num callback objects: %d" << m_CallbackObjects.size());
  // }

  // shared_ptr handles this for us now
  // envSTLHelpers::DeleteContainer(this->m_CallbackObjects);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
const std::string&
prtyProperty::GetPropertyName() const
{
  return m_Name;
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
void
prtyProperty::SetPropertyName(const std::string& i_Name)
{
  m_Name = i_Name;
}

//--------------------------------------------------------------------
// Animatable - can this property be animated?
//--------------------------------------------------------------------
bool
prtyProperty::IsAnimatable() const
{
  return m_Flags.m_bAnimatable;
}
void
prtyProperty::SetAnimatable(bool i_bVal)
{
  m_Flags.m_bAnimatable = i_bVal;
}

//--------------------------------------------------------------------
// Scripted - Is this property currently being animated?
//--------------------------------------------------------------------
bool
prtyProperty::IsScripted() const
{
  return m_Flags.m_bScripted;
}
void
prtyProperty::SetScripted(bool i_bVal)
{
  m_Flags.m_bScripted = i_bVal;
}

//--------------------------------------------------------------------
// ValueVariation - Does the current value vary from the scripted value?
//--------------------------------------------------------------------
bool
prtyProperty::HasValueVariation() const
{
  return m_Flags.m_bValueVariation;
}
void
prtyProperty::SetValueVariation(bool i_bVal)
{
  m_Flags.m_bValueVariation = i_bVal;
}

//--------------------------------------------------------------------
//	assignment operator - does not copy the callbacks
//--------------------------------------------------------------------
prtyProperty&
prtyProperty::operator=(const prtyProperty& i_Property)
{
  // m_Type = i_Property.m_Type;
  m_Name = i_Property.m_Name;

  // flags should not be copied
  // m_Flags = i_Property.m_Flags;

  return *this;
}

//--------------------------------------------------------------------
// AddCallback using interface.
//
//	NOTE: This property will take ownership of the callback
//	pointer and will delete it in its destructor! It should be
//	created on the heap and probably should use the
//	prtyCallbackWrapper template.
//--------------------------------------------------------------------
void
prtyProperty::AddCallback(prtyPropertyCallback* i_pCallback)
{
  if (i_pCallback != NULL)
    m_CallbackObjects.push_back(std::shared_ptr<prtyPropertyCallback>(i_pCallback));
}

//--------------------------------------------------------------------
// AddCallback shared pointer variation for when you want to
//	keep a copy of the callback also.
//--------------------------------------------------------------------
void
prtyProperty::AddCallback(std::shared_ptr<prtyPropertyCallback> i_pCallback)
{
  if (i_pCallback.get() != NULL)
    m_CallbackObjects.push_back(i_pCallback);
}

//--------------------------------------------------------------------
// RemoveCallback by shared_ptr. You should be keeping a shared_ptr
//	to the callback passed to AddCallback() if you want to remove
//	it later. Don't use the direct pointer function when adding
//	if you want to keep a copy.
//--------------------------------------------------------------------
void
prtyProperty::RemoveCallback(std::shared_ptr<prtyPropertyCallback> i_pCallback)
{
  envSTLHelpers::RemoveOneValue(m_CallbackObjects, i_pCallback);
}

//--------------------------------------------------------------------
//	call all the callbacks
//--------------------------------------------------------------------
void
prtyProperty::NotifyCallbacksPropertyChanged(bool i_bWithUndo)
{
  const int num_callbackobjs = m_CallbackObjects.size();
  for (int i = 0; i < num_callbackobjs; ++i) {
    m_CallbackObjects[i]->PropertyChanged(this, i_bWithUndo);
  }
}

//--------------------------------------------------------------------
//  In some cases we want the form builder to check whether or not
// this property should be visible.
//--------------------------------------------------------------------
void
prtyProperty::SetVisible(bool i_bVisible)
{
  m_bVisible = i_bVisible;
}

bool
prtyProperty::GetVisible() const
{
  return m_bVisible;
}

//--------------------------------------------------------------------
//	The type of property it is
//--------------------------------------------------------------------
// const std::string& prtyProperty::GetType()
//{
//	return m_Type;
//}

//--------------------------------------------------------------------
//	The type of property it is
//--------------------------------------------------------------------
// void prtyProperty::SetType(const std::string i_Type)
//{
//	m_Type = i_Type;
//}
