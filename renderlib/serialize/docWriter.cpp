#include "docWriter.h"

#include "core/prty/prtyProperty.hpp"
#include "core/prty/prtyObject.hpp"

// assumes a current named object context (i.e. beginObject was already called).
// if objects had ONLY properties, and no children,
// then this could create the object context itself using beginObject/endObject.
void
docWriter::writeProperties(prtyObject* obj)
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
        // std::cout << "Property is null!" << std::endl;
        continue;
      }

      writePrty(prop);
    }
  }
}
