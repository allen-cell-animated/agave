/****************************************************************************\
**	prtyRotation.hpp
**
**		Rotation property
**
**
**
\****************************************************************************/
#ifdef PRTY_ROTATION_HPP
#error prtyRotation.hpp multiply included
#endif
#define PRTY_ROTATION_HPP

#ifndef PRTY_PROPERTY_HPP
#include "core/prty/prtyProperty.hpp"
#endif

#include "glm.h"

#include <string>

//============================================================================
//============================================================================
class prtyRotation : public prtyProperty
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyRotation();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyRotation(const std::string& i_Name);

  //--------------------------------------------------------------------
  // Constructor with quaternion initial value
  //--------------------------------------------------------------------
  prtyRotation(const std::string& i_Name, const glm::quat& i_InitialValue);

  //--------------------------------------------------------------------
  // Constructor with euler angle initial value
  //--------------------------------------------------------------------
  prtyRotation(const std::string& i_Name, float i_X, float i_Y, float i_Z);

  //--------------------------------------------------------------------
  //	The type of property it is
  //--------------------------------------------------------------------
  virtual const char* GetType();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  const glm::quat& GetQuaternion() const;
  const glm::quat& GetValue() const;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  void SetQuaternion(const glm::quat& i_Rotation, bool i_bDirty = false); // UndoFlags i_Undoable = eNoUndo);
  void SetValue(const glm::quat& i_Rotation, bool i_bDirty = false);      // UndoFlags i_Undoable = eNoUndo);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  // const glm::quat& GetMinimum() const;
  // void SetMinimum(const glm::quat& i_Minimum);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  // const glm::quat& GetMaximum() const;
  // void SetMaximum(const glm::quat& i_Maximum);

  //--------------------------------------------------------------------
  //	Set rotation through 3 euler angles (in radians)
  //--------------------------------------------------------------------
  void SetEuler(float i_X, float i_Y, float i_Z, bool i_bDirty = false); // UndoFlags i_Undoable = eNoUndo);
  void GetEuler(float& o_X, float& o_Y, float& o_Z) const;

  //--------------------------------------------------------------------
  //	operators
  //--------------------------------------------------------------------
  prtyRotation& operator=(const glm::quat& i_Value);
  prtyRotation& operator=(const prtyRotation& i_Value);

  //--------------------------------------------------------------------
  //	comparison operators
  //--------------------------------------------------------------------
  bool operator==(const glm::quat& i_Value) const;
  bool operator==(const prtyRotation& i_Value) const;
  bool operator!=(const prtyRotation& i_Value) const;

  //--------------------------------------------------------------------
  // Create an undo operation of the correct type for this
  // property. A reference to this property should be passed in.
  //--------------------------------------------------------------------
  virtual undoUndoOperation* CreateUndoOperation(std::shared_ptr<prtyPropertyReference> i_pPropertyRef);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Read(chReader& io_Reader);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Write(chWriter& io_Writer) const;

private:
  glm::quat m_Quaternion;
  glm::vec3 m_EulerAngles;
  // glm::quat	m_Minimum;
  // glm::quat	m_Maximum;
};
