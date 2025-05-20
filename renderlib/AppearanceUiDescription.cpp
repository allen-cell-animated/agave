#include "AppearanceUiDescription.hpp"

ComboBoxUiInfo AppearanceUiDescription::m_rendererType("Renderer Type",
                                                       "Select volume rendering type",
                                                       "Select volume rendering type",
                                                       { "Ray march blending", "Path Traced" });
ComboBoxUiInfo AppearanceUiDescription::m_shadingType("Shading Type",
                                                      "Select volume shading style",
                                                      "Select volume shading style",
                                                      { "BRDF Only", "Phase Function Only", "Mixed" });
FloatSliderSpinnerUiInfo AppearanceUiDescription::m_densityScale("Scattering Density",
                                                                 "Set scattering density for volume",
                                                                 "Set scattering density for volume",
                                                                 0.001f,
                                                                 100.0f,
                                                                 3,
                                                                 0.01f,
                                                                 10);
FloatSliderSpinnerUiInfo AppearanceUiDescription::m_gradientFactor("Shading Type Mixture",
                                                                   "Mix between BRDF and Phase shading",
                                                                   "Mix between BRDF and Phase shading",
                                                                   0.0f,
                                                                   1.0f,
                                                                   3,
                                                                   0.01f,
                                                                   10);
FloatSliderSpinnerUiInfo AppearanceUiDescription::m_stepSizePrimaryRay("Primary Ray Step Size",
                                                                       "Set volume ray march step size for camera rays",
                                                                       "Set volume ray march step size for camera rays",
                                                                       1.0f,
                                                                       100.0f,
                                                                       3,
                                                                       0.01f,
                                                                       10);
FloatSliderSpinnerUiInfo AppearanceUiDescription::m_stepSizeSecondaryRay(
  "Secondary Ray Step Size",
  "Set volume ray march step size for scattered rays",
  "Set volume ray march step size for scattered rays",
  1.0f,
  100.0f,
  3,
  0.01f,
  10);
CheckBoxUiInfo AppearanceUiDescription::m_interpolate("Interpolate",
                                                      "Interpolated volume sampling",
                                                      "Interpolated volume sampling");
ColorPickerUiInfo AppearanceUiDescription::m_backgroundColor("Background Color",
                                                             "Set background color",
                                                             "Set background color");
CheckBoxUiInfo AppearanceUiDescription::m_showBoundingBox("Show Bounding Box",
                                                          "Show/hide bounding box",
                                                          "Show/hide bounding box");
ColorPickerUiInfo AppearanceUiDescription::m_boundingBoxColor("Bounding Box Color",
                                                              "Set bounding box color",
                                                              "Set bounding box color");
CheckBoxUiInfo AppearanceUiDescription::m_showScaleBar("Show Scale Bar", "Show/hide scale bar", "Show/hide scale bar");
