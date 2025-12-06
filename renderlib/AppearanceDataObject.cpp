#include "AppearanceDataObject.hpp"

#include "Enumerations.h"
#include "Logging.h"

AppearanceDataObject::AppearanceDataObject()
{
  RendererType.SetEnumTag(0, "Ray march blending");
  RendererType.SetEnumTag(1, "Path Traced");

  ShadingType.SetEnumTag(0, "BRDF Only");
  ShadingType.SetEnumTag(1, "Phase Function Only");
  ShadingType.SetEnumTag(2, "Mixed");
}
