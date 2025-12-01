#include "docReader.h"

#include "core/prty/prtyProperty.hpp"
#include "core/prty/prtyObject.hpp"

// Reads all properties from the current object context
// Assumes we're already inside the object (after beginObject was called)
void
docReader::readProperties(prtyObject* obj)
{
  if (!obj) {
    return;
  }

  const PropertyUIIList& propList = obj->GetList();

  for (const auto& propUIInfo : propList) {
    int numProps = propUIInfo->GetNumberOfProperties();

    for (int i = 0; i < numProps; ++i) {
      prtyProperty* prop = propUIInfo->GetProperty(i);
      if (!prop) {
        continue;
      }

      std::string propName = prop->GetPropertyName();

      // Check if the property exists in the document
      if (hasKey(propName.c_str())) {
        // Set up the property name for reading and let the property read itself
        readPrty(prop);
      }
    }
  }
}
