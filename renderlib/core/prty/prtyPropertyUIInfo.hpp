/****************************************************************************\
**	prtyPropertyUIInfo.hpp
**
**		Property UI info
**
**
**
\****************************************************************************/
#ifdef PRTY_PROPERTYUIINFO_HPP
#error prtyPropertyUIInfo.hpp multiply included
#endif
#define PRTY_PROPERTYUIINFO_HPP

#include <string>
#include <vector>

//============================================================================
//============================================================================
class prtyProperty;
class prtyReferenceCreator;

// TODO add statusTip and toolTip to prtyPropertyUIInfo
// struct GenericUIInfo
// {
//   std::string type;
//   std::string formLabel;
//   std::string statusTip;
//   std::string toolTip;

//   GenericUIInfo() = default;
//   GenericUIInfo(std::string type, std::string formLabel, std::string statusTip, std::string toolTip)
//     : type(type)
//     , formLabel(formLabel)
//     , statusTip(statusTip)
//     , toolTip(toolTip)
//   {
//   }
// };

//============================================================================
//============================================================================
class prtyPropertyUIInfo
{
public:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyPropertyUIInfo(prtyProperty* i_pProperty);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyPropertyUIInfo(prtyProperty* i_pProperty, const std::string& i_Category, const std::string& i_Description);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  virtual ~prtyPropertyUIInfo();

  //--------------------------------------------------------------------
  // Return pointer to new equivalent prtyPropertyUIInfo
  //--------------------------------------------------------------------
  virtual prtyPropertyUIInfo* Clone();

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  int GetNumberOfProperties() const;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  prtyProperty* GetProperty(int i_Index) const;
  prtyProperty* GetProperty(const std::string& i_Name) const;

  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------
  prtyReferenceCreator* GetReferenceCreator(int i_Index) const;

  //--------------------------------------------------------------------
  // Add property to list, optionally passing a reference creator
  // that can be used when creating undo operations.
  //--------------------------------------------------------------------
  void AddProperty(prtyProperty* i_pProperty, prtyReferenceCreator* i_pReferenceCreator = NULL);

  //--------------------------------------------------------------------
  // Can be used to assign a new reference creator to all existing
  //	properties in this UIInfo. This will not affect new properties
  //	added after this call.
  //--------------------------------------------------------------------
  void AssignReferenceCreator(prtyReferenceCreator* i_pReferenceCreator);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  const std::string& GetControlName() const;

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  const std::string& GetCategory() const;
  void SetCategory(const std::string& i_Category);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  const std::string& GetDescription() const;
  void SetDescription(const std::string& i_Description);

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  const bool GetReadOnly() const;
  void SetReadOnly(const bool i_bReadOnly);

  //--------------------------------------------------------------------
  // ConfirmationString is message to display to user that he needs
  // to confirm before the value of the property can be changed.
  //--------------------------------------------------------------------
  void SetConfirmationString(const std::string& i_Confirmation);
  const std::string& GetConfirmationString() const;

  //--------------------------------------------------------------------
  //	Call this function when the property UI Info has been changed
  //	after a control for it has already been created. It will
  //	cause the control to be updated.
  //	Note: Con't call this automatically from derived classes,
  //	just let the prtyObject that owns this UI Info handle it.
  //--------------------------------------------------------------------
  void UpdateControl();

protected:
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  void SetControlName(const std::string i_ControlName);

private:
  std::vector<prtyProperty*> m_Properties;
  std::vector<prtyReferenceCreator*> m_ReferenceCreators;

  std::string m_ControlName;
  std::string m_Category;
  std::string m_Description;
  bool m_bReadOnly;
  std::string m_ConfirmationString;
};
