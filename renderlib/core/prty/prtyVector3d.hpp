#pragma once

#include "core/prty/prtyPropertyTemplate.hpp"

#include "glm.h"

//============================================================================
//============================================================================
class prtyVector3d : public prtyPropertyTemplate<glm::vec3, const glm::vec3&>
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyVector3d();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyVector3d(const std::string& i_Name);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyVector3d(const std::string& i_Name, const glm::vec3& i_InitialValue);

  //--------------------------------------------------------------------
  //	The type of property it is
  //--------------------------------------------------------------------
  virtual const char* GetType();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  void Set(const float i_ValueX,
           const float i_ValueY,
           const float i_ValueZ,
           bool i_bDirty = false); // UndoFlags i_Undoable = eNoUndo );

  //--------------------------------------------------------------------
  // Set whether this property should consider the current units
  // when displaying its value in the user interface.
  // prtyVector3d has a default value of false.
  //--------------------------------------------------------------------
  bool GetUseUnits() const;
  void SetUseUnits(bool i_bUseUnits);

  //--------------------------------------------------------------------
  // Get and Set value in the current display units.
  // Converts to internal units and then calls Get/SetValue().
  // If this property does not use units, the unscaled value is used.
  //--------------------------------------------------------------------
  glm::vec3 GetScaledValue() const;
  void SetScaledValue(const glm::vec3& i_Value, bool i_bDirty = false); // UndoFlags i_Undoable = eNoUndo);

  //--------------------------------------------------------------------
  //	operators
  //--------------------------------------------------------------------
  prtyVector3d& operator=(const glm::vec3& i_Value);
  prtyVector3d& operator=(const prtyVector3d& i_Value);

  //--------------------------------------------------------------------
  //	comparison operators
  //--------------------------------------------------------------------
  bool operator==(const prtyVector3d& i_Property) const;
  bool operator!=(const prtyVector3d& i_Property) const;
  bool operator==(const glm::vec3& i_Value) const;
  bool operator!=(const glm::vec3& i_Value) const;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Read(chReader& io_Reader);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual void Write(docWriter& io_Writer) const;

private:
  bool m_bUseUnits;
};
