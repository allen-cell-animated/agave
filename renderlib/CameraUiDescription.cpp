#include "CameraUiDescription.hpp"

FloatSliderSpinnerUiInfo CameraUiDescription::m_exposure("Exposure",
                                                         "Set Exposure",
                                                         "Set camera exposure",
                                                         0.0f,
                                                         1.0f,
                                                         2,    // decimals
                                                         0.01, // singleStep
                                                         0     // numTickMarks
);
ComboBoxUiInfo CameraUiDescription::m_exposureIterations("Exposure Time",
                                                         "Set Exposure Time",
                                                         "Set number of samples to accumulate per viewport update",
                                                         { "1", "2", "4", "8" });
CheckBoxUiInfo CameraUiDescription::m_noiseReduction("Noise Reduction",
                                                     "Enable denoising pass",
                                                     "Enable denoising pass");
FloatSliderSpinnerUiInfo CameraUiDescription::m_apertureSize("Aperture Size",
                                                             "Set camera aperture size",
                                                             "Set camera aperture size",
                                                             0.0f,
                                                             0.1f,
                                                             2,    // decimals
                                                             0.01, // singleStep
                                                             0,    // numTickMarks
                                                             " mm");
FloatSliderSpinnerUiInfo CameraUiDescription::m_fieldOfView("Field of view",
                                                            "Set camera field of view angle",
                                                            "Set camera field of view angle",
                                                            10.0f,
                                                            150.0f,
                                                            2,    // decimals
                                                            0.01, // singleStep
                                                            0,    // numTickMarks
                                                            " deg");
FloatSliderSpinnerUiInfo CameraUiDescription::m_focalDistance("Focal distance",
                                                              "Set focal distance",
                                                              "Set focal distance",
                                                              0.0f,
                                                              15.0f,
                                                              2,    // decimals
                                                              0.01, // singleStep
                                                              0,    // numTickMarks
                                                              " m");
