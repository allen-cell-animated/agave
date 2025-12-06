#pragma once

#include <string>

//============================================================================
//============================================================================
class prtyInterest
{
public:
  //------------------------------------------------------------------------
  // Constructor/Destructor
  //------------------------------------------------------------------------
  prtyInterest(std::string i_Name,
               void (*i_CallbackFunction)() = NULL,
               bool i_bEditable = true,
               bool i_bProcedural = false);
  ~prtyInterest();

  //------------------------------------------------------------------------
  // Is this interest editable in a control
  //------------------------------------------------------------------------
  bool IsEditable();
  void SetEditable(bool i_bEditable);

  //------------------------------------------------------------------------
  // Is this interest a procedural texture, this will override any pre-existing
  // data in the control
  //------------------------------------------------------------------------
  bool IsProcedural();
  void SetProcedural(bool i_bProcedural);

  //------------------------------------------------------------------------
  // Get the name of the interest
  //------------------------------------------------------------------------
  std::string& GetName();
  void SetName(std::string& i_Name);

  //------------------------------------------------------------------------
  // Execute the interest's callback function
  //------------------------------------------------------------------------
  void Invoke();

private:
  bool m_bEditable;
  bool m_bProcedural;
  std::string m_Name;
  void (*m_CallbackFunction)();

}; // end prtyInterest class
