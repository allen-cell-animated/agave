#include "docWriter.h"

#include "core/prty/prtyProperty.hpp"
#include "core/prty/prtyObject.hpp"

// assumes a current named object context.
// if objects had ONLY properties, and no children,
// then this could create the object context itself using beginObject/endObject.
void
docWriter::writeProperties(prtyObject* obj)
{
  if (!obj) {
    return;
  }

  // beginObject(name.c_str());

  const PropertyUIIList& propList = obj->GetList();
  // std::cout << "Property list size: " << propList.size() << std::endl;

  for (const auto& propUIInfo : propList) {
    int numProps = propUIInfo->GetNumberOfProperties();
    // std::cout << "Number of properties in UIInfo: " << numProps << std::endl;

    for (int i = 0; i < numProps; ++i) {
      prtyProperty* prop = propUIInfo->GetProperty(i);
      if (!prop) {
        // std::cout << "Property is null!" << std::endl;
        continue;
      }

      const char* type = prop->GetType();
      std::string propName = prop->GetPropertyName();
      // std::cout << "Writing property: " << propName << " of type: " << type << std::endl;

      // Set up the property name for writing
      writePrty(prop);
    }
  }

  // endObject();
}
