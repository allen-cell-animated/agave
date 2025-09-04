#pragma once

#include <memory>

//============================================================================
//	Forward References
//============================================================================
class prtyProperty;

//============================================================================
// A prtyReference is a logical reference to a property that may point to
//	different actual prtyProperty classes at different times. This may happen
//	during undo and redo when an object is deleted and restored.

// Provides a safe way to refer to a property for undo operations.
// Derivations of this class should provide a way to access a property
// by a reference that is consistent through deletion and restoration of
// the parent property object.
//============================================================================
class prtyPropertyReference
{
public:
  //--------------------------------------------------------------------
  // virtual destructor
  //--------------------------------------------------------------------
  virtual ~prtyPropertyReference() {}

  //--------------------------------------------------------------------
  //	GetProperty - return a pointer to a property that is usable
  //	for a short period of time.
  //--------------------------------------------------------------------
  virtual prtyProperty* GetProperty() = 0;
};

//============================================================================
// prtyReferenceCreator creates logical references to properties
//============================================================================
class prtyReferenceCreator
{
public:
  //--------------------------------------------------------------------
  //	CreateReferenceForProperty - given a property, create a
  //	shared_ptr to a prtyPropertyReference to this property.
  //--------------------------------------------------------------------
  virtual std::shared_ptr<prtyPropertyReference> CreateReferenceForProperty(prtyProperty& i_Property) = 0;
};
