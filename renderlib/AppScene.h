#pragma once

#include "BoundingBox.h"
#include "Colormap.h"
#include "Defines.h"
#include "DenoiseParams.h"
#include "GradientData.h"
#include "Light.h"
#include "Lighting.h"
#include "Object3d.h"
#include "ScenePlane.h"
#include "SceneLight.h"
#include "Timeline.h"
#include "glm.h"

#include <memory>
#include <vector>

class ImageXYZC;

static constexpr size_t MAX_CPU_CHANNELS = 32;
static constexpr size_t MAX_CLIP_PLANES = 4;

struct VolumeDisplay
{
  bool m_showBoundingBox = true;
  float m_boundingBoxColor[3] = { 1.0f, 1.0f, 1.0f };
  float m_backgroundColor[3] = { 0.0f, 0.0f, 0.0f };
  float m_DensityScale = 50.0f;
  float m_GradientFactor = 0.1f;
  int m_ShadingType = 2;

  // channels enabled/disabled
  // per channel colors
  float m_diffuse[MAX_CPU_CHANNELS * 3];
  float m_specular[MAX_CPU_CHANNELS * 3];
  float m_emissive[MAX_CPU_CHANNELS * 3];
  float m_roughness[MAX_CPU_CHANNELS];
  float m_opacity[MAX_CPU_CHANNELS];
  bool m_enabled[MAX_CPU_CHANNELS];
  float m_labels[MAX_CPU_CHANNELS];
  ColorRamp m_colormap[MAX_CPU_CHANNELS];

  GradientData m_gradientData[MAX_CPU_CHANNELS];

  // Per-channel clip plane group assignment.
  // Index into Scene::m_clipPlanes. -1 means no clip plane for that channel.
  int32_t m_clipPlaneGroup[MAX_CPU_CHANNELS];

  VolumeDisplay();
};

class Scene
{
public:
  Timeline m_timeLine;

  // one single volume, for now...!
  std::shared_ptr<ImageXYZC> m_volume;
  // appearance settings for a volume
  VolumeDisplay m_material;

  CBoundingBox m_roi = CBoundingBox(glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));
  std::shared_ptr<ScenePlane> m_clipPlanes[MAX_CLIP_PLANES];
  // Backward-compatible alias for m_clipPlanes[0]
  std::shared_ptr<ScenePlane>& m_clipPlane = m_clipPlanes[0];

  Lighting m_lighting;

  // convenience functions
  // For now, this must match the order in which the lights were added, in initLights
  static constexpr int SphereLightIndex = 0;
  Light& SphereLight() const { return *m_lighting.m_Lights[SphereLightIndex]; }
  SceneLight* SceneSphereLight() const { return m_lighting.m_sceneLights[SphereLightIndex]; }
  static constexpr int AreaLightIndex = 1;
  Light& AreaLight() const { return *m_lighting.m_Lights[AreaLightIndex]; }
  SceneLight* SceneAreaLight() const { return m_lighting.m_sceneLights[AreaLightIndex]; }

  // weak ptr! must not outlive the objects it points to.
  SceneObject* m_selection = nullptr;

  CBoundingBox m_boundingBox;
  bool m_showScaleBar = false;
  bool m_showAxisHelper = false;

  void initLights();
  void initSceneFromImg(std::shared_ptr<ImageXYZC> img);
  void initBounds(const CBoundingBox& bb);
  void initBoundsFromImg(std::shared_ptr<ImageXYZC> img);
  void getFirst4EnabledChannels(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3) const;

  void setLockLightsToCamera(bool lock) { m_lighting.lockToCamera = lock; }
};
