#pragma once

// Single source of truth for the set of wire-protocol commands.
//
// Add a new command in ONE place here; it will automatically be:
//   - dispatched in agave_app/commandBuffer.cpp
//   - covered by the command-registry unit tests (ID + python-name uniqueness,
//     and an assertion that the binary dispatcher recognizes the ID).
//
// The per-command round-trip test (write -> parse -> toPythonString) still
// needs to be written by hand with representative data, since only the author
// knows what valid field values look like.

#define AGAVE_COMMAND_LIST(X)                                                                                          \
  X(SessionCommand)                                                                                                    \
  X(AssetPathCommand)                                                                                                  \
  X(LoadOmeTifCommand)                                                                                                 \
  X(SetCameraPosCommand)                                                                                               \
  X(SetCameraTargetCommand)                                                                                            \
  X(SetCameraUpCommand)                                                                                                \
  X(SetCameraApertureCommand)                                                                                          \
  X(SetCameraProjectionCommand)                                                                                        \
  X(SetCameraFocalDistanceCommand)                                                                                     \
  X(SetCameraExposureCommand)                                                                                          \
  X(SetDiffuseColorCommand)                                                                                            \
  X(SetSpecularColorCommand)                                                                                           \
  X(SetEmissiveColorCommand)                                                                                           \
  X(SetRenderIterationsCommand)                                                                                        \
  X(SetStreamModeCommand)                                                                                              \
  X(RequestRedrawCommand)                                                                                              \
  X(SetResolutionCommand)                                                                                              \
  X(SetDensityCommand)                                                                                                 \
  X(FrameSceneCommand)                                                                                                 \
  X(SetGlossinessCommand)                                                                                              \
  X(EnableChannelCommand)                                                                                              \
  X(SetWindowLevelCommand)                                                                                             \
  X(OrbitCameraCommand)                                                                                                \
  X(SetSkylightTopColorCommand)                                                                                        \
  X(SetSkylightMiddleColorCommand)                                                                                     \
  X(SetSkylightBottomColorCommand)                                                                                     \
  X(SetLightPosCommand)                                                                                                \
  X(SetLightColorCommand)                                                                                              \
  X(SetLightSizeCommand)                                                                                               \
  X(SetClipRegionCommand)                                                                                              \
  X(SetVoxelScaleCommand)                                                                                              \
  X(AutoThresholdCommand)                                                                                              \
  X(SetPercentileThresholdCommand)                                                                                     \
  X(SetOpacityCommand)                                                                                                 \
  X(SetPrimaryRayStepSizeCommand)                                                                                      \
  X(SetSecondaryRayStepSizeCommand)                                                                                    \
  X(SetBackgroundColorCommand)                                                                                         \
  X(SetIsovalueThresholdCommand)                                                                                       \
  X(SetControlPointsCommand)                                                                                           \
  X(LoadVolumeFromFileCommand)                                                                                         \
  X(SetTimeCommand)                                                                                                    \
  X(SetBoundingBoxColorCommand)                                                                                        \
  X(ShowBoundingBoxCommand)                                                                                            \
  X(TrackballCameraCommand)                                                                                            \
  X(LoadDataCommand)                                                                                                   \
  X(ShowScaleBarCommand)                                                                                               \
  X(SetFlipAxisCommand)                                                                                                \
  X(SetInterpolationCommand)                                                                                           \
  X(SetClipPlaneCommand)                                                                                               \
  X(SetColorRampCommand)                                                                                               \
  X(SetMinMaxThresholdCommand)                                                                                         \
  X(SetSkylightRotationCommand)                                                                                        \
  X(SetClipPlaneIndexCommand)                                                                                          \
  X(EnableClipPlaneCommand)                                                                                            \
  X(SetChannelClipPlaneGroupCommand)
