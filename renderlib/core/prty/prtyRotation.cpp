#include "core/prty/prtyRotation.hpp"

#include "core/prty/prtyRotationUndo.hpp"

// #include "core/ch/chChunkParserUtil.hpp"
// #include "core/ch/chReader.hpp"
// #include "core/ma/maConstants.hpp"
#include "core/undo/undoUndoMgr.hpp"

namespace {
const float c_fEpsilon = 1.0e-6f;

void
update_euler_from_quaternion(const glm::quat& i_Quaternion, glm::vec3& o_EulerAngles)
{
  o_EulerAngles = glm::eulerAngles(i_Quaternion) * 3.14159f / 180.f;
}

void
update_quaternion_from_euler(const glm::vec3& i_EulerAngles, glm::quat& o_Quaternion)
{
  o_Quaternion = glm::quat(glm::radians(i_EulerAngles));
}
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyRotation::prtyRotation()
  : prtyProperty("Rotation")
{
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyRotation::prtyRotation(const std::string& i_Name)
  : prtyProperty(i_Name)
{
}

//--------------------------------------------------------------------
// Constructor with quaternion initial value
//--------------------------------------------------------------------
prtyRotation::prtyRotation(const std::string& i_Name, const glm::quat& i_InitialValue)
  : prtyProperty(i_Name)
  , m_Quaternion(i_InitialValue)
{
  update_euler_from_quaternion(m_Quaternion, m_EulerAngles);
}

//--------------------------------------------------------------------
// Constructor with euler angle initial value
//--------------------------------------------------------------------
prtyRotation::prtyRotation(const std::string& i_Name, float i_X, float i_Y, float i_Z)
  : m_EulerAngles(i_X, i_Y, i_Z)
{
  update_quaternion_from_euler(m_EulerAngles, m_Quaternion);
}

//--------------------------------------------------------------------
//	The type of property it is
//--------------------------------------------------------------------
const char*
prtyRotation::GetType()
{
  return "Rotation";
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
const glm::quat&
prtyRotation::GetQuaternion() const
{
  return m_Quaternion;
}
const glm::quat&
prtyRotation::GetValue() const
{
  return m_Quaternion;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void
prtyRotation::SetValue(const glm::quat& i_Rotation, bool i_bDirty) // UndoFlags i_Undoable)
{
  this->SetQuaternion(i_Rotation, i_bDirty); // i_Undoable);
}
void
prtyRotation::SetQuaternion(const glm::quat& i_Rotation, bool i_bDirty) // UndoFlags i_Undoable)
{
  if (i_Rotation != m_Quaternion) {
    m_Quaternion = i_Rotation;
    update_euler_from_quaternion(m_Quaternion, m_EulerAngles);
    // NotifyCallbacksPropertyChanged(i_Undoable != eNoUndo);
    NotifyCallbacksPropertyChanged(i_bDirty);
  }
}

//--------------------------------------------------------------------
//	Set rotation through 3 euler angles
//--------------------------------------------------------------------
void
prtyRotation::SetEuler(float i_X, float i_Y, float i_Z, bool i_bDirty) // UndoFlags i_Undoable)
{
  glm::vec3 euler_angles(i_X, i_Y, i_Z);
  if (euler_angles != m_EulerAngles) {
    m_EulerAngles = euler_angles;
    update_quaternion_from_euler(m_EulerAngles, m_Quaternion);
    // NotifyCallbacksPropertyChanged(i_Undoable != eNoUndo);
    NotifyCallbacksPropertyChanged(i_bDirty);
  }
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
void
prtyRotation::GetEuler(float& o_X, float& o_Y, float& o_Z) const
{
  o_X = m_EulerAngles.x;
  o_Y = m_EulerAngles.y;
  o_Z = m_EulerAngles.z;
}

//--------------------------------------------------------------------
//	operators
//--------------------------------------------------------------------
prtyRotation&
prtyRotation::operator=(const glm::quat& i_Value)
{
  SetQuaternion(i_Value);
  return *this;
}
prtyRotation&
prtyRotation::operator=(const prtyRotation& i_Value)
{
  // copy base data
  prtyProperty::operator=(i_Value);

  m_Quaternion = i_Value.m_Quaternion;
  m_EulerAngles = i_Value.m_EulerAngles;
  // SetMinimum(i_Value.GetMinimum());
  // SetMaximum(i_Value.GetMaximum());

  // Notify callbacks, since we set the values directly instead of using
  // SetValue()-like functions
  const bool bDirty = false;
  NotifyCallbacksPropertyChanged(bDirty);

  return *this;
}

//--------------------------------------------------------------------
//	comparison operators
//--------------------------------------------------------------------
bool
prtyRotation::operator==(const glm::quat& i_Value) const
{
  return (m_Quaternion == i_Value);
}
bool
prtyRotation::operator==(const prtyRotation& i_Value) const
{
  // Should this compare quaternion or euler angles?
  // return (m_Quaternion == i_Value.m_Quaternion);

  // comparison wihtin epsilon
  return (glm::length2(m_EulerAngles - i_Value.m_EulerAngles) < c_fEpsilon);
}
bool
prtyRotation::operator!=(const prtyRotation& i_Value) const
{
  // Should this compare quaternion or euler angles?
  // return !(m_Quaternion == i_Value.m_Quaternion);

  // comparison wihtin epsilon
  return (glm::length2(m_EulerAngles - i_Value.m_EulerAngles) >= c_fEpsilon);
}

//--------------------------------------------------------------------
// Create an undo operation of the correct type for this
// property. A reference to this property should be passed in.
//--------------------------------------------------------------------
// virtual
undoUndoOperation*
prtyRotation::CreateUndoOperation(std::shared_ptr<prtyPropertyReference> i_pPropertyRef)
{
  return new prtyRotationUndo(i_pPropertyRef, m_EulerAngles);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyRotation::Read(chReader& io_Reader)
{
  // // We need to write euler angles, how to handle versions?
  // glm::quat temp;
  // chChunkParserUtil::Read(io_Reader, temp);
  // SetQuaternion(temp);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyRotation::Write(chWriter& io_Writer) const
{
  // chChunkParserUtil::Write(io_Writer, m_Quaternion);
}
