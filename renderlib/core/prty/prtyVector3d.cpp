/****************************************************************************\
**	prtyVector3d.cpp
**
**		see .hpp
**
**
**
\****************************************************************************/
#include "core/prty/prtyVector3d.hpp"
#include "core/prty/prtyUnits.hpp"

// #include "core/ch/chChunkParserUtil.hpp"
// #include "core/ch/chReader.hpp"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyVector3d::prtyVector3d()
  : prtyPropertyTemplate("Point3d", glm::vec3(0, 0, 0))
  , m_bUseUnits(false)
{
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyVector3d::prtyVector3d(const std::string& i_Name)
  : prtyPropertyTemplate(i_Name, glm::vec3(0, 0, 0))
  , m_bUseUnits(false)
{
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
prtyVector3d::prtyVector3d(const std::string& i_Name, const glm::vec3& i_InitialValue)
  : prtyPropertyTemplate(i_Name, i_InitialValue)
  , m_bUseUnits(false)
{
}

//--------------------------------------------------------------------
//	The type of property it is
//--------------------------------------------------------------------
const char*
prtyVector3d::GetType()
{
  return "Vector3d";
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
void
prtyVector3d::Set(const float i_ValueX,
                  const float i_ValueY,
                  const float i_ValueZ,
                  bool i_bDirty) // UndoFlags i_Undoable )
{
  glm::vec3 newvec(i_ValueX, i_ValueY, i_ValueZ);
  SetValue(newvec, i_bDirty); // i_Undoable);
}

//--------------------------------------------------------------------
//	operators
//--------------------------------------------------------------------
prtyVector3d&
prtyVector3d::operator=(const glm::vec3& i_Value)
{
  SetValue(i_Value);
  return *this;
}
prtyVector3d&
prtyVector3d::operator=(const prtyVector3d& i_Property)
{
  // copy base data
  prtyProperty::operator=(i_Property);
  SetValue(i_Property.GetValue());
  m_bUseUnits = i_Property.m_bUseUnits;
  return *this;
}

//--------------------------------------------------------------------
//	comparison operators
//--------------------------------------------------------------------
bool
prtyVector3d::operator==(const prtyVector3d& i_Property) const
{
  return (m_Value == i_Property.GetValue());
}
bool
prtyVector3d::operator!=(const prtyVector3d& i_Property) const
{
  return (m_Value != i_Property.GetValue());
}
bool
prtyVector3d::operator==(const glm::vec3& i_Value) const
{
  return (m_Value == i_Value);
}
bool
prtyVector3d::operator!=(const glm::vec3& i_Value) const
{
  return (m_Value != i_Value);
}

//--------------------------------------------------------------------
// Set whether this property should consider the current units
// when displaying its value in the user interface.
//--------------------------------------------------------------------
bool
prtyVector3d::GetUseUnits() const
{
  return m_bUseUnits;
}
void
prtyVector3d::SetUseUnits(bool i_bUseUnits)
{
  m_bUseUnits = true;
}

//--------------------------------------------------------------------
// Get and Set value in the current display units.
// Converts to internal units and then calls Get/SetValue()
//--------------------------------------------------------------------
glm::vec3
prtyVector3d::GetScaledValue() const
{
  if (this->GetUseUnits())
    return (this->GetValue() / prtyUnits::GetUnitScaling());
  else
    return this->GetValue();
}
void
prtyVector3d::SetScaledValue(const glm::vec3& i_Value, bool i_bDirty) // UndoFlags i_Undoable)
{
  if (this->GetUseUnits())
    this->SetValue(i_Value * prtyUnits::GetUnitScaling(), i_bDirty); // i_Undoable);
  else
    return this->SetValue(i_Value, i_bDirty); // i_Undoable);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyVector3d::Read(chReader& io_Reader)
{
  // glm::vec3 temp;
  // chChunkParserUtil::Read(io_Reader, temp);
  // SetValue(temp);
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------
// virtual
void
prtyVector3d::Write(chWriter& io_Writer) const
{
  // chChunkParserUtil::Write(io_Writer, GetValue());
}
