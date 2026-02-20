#include "ViewerWindow.h"

#include "AppScene.h"
#include "AxisHelperTool.h"
#include "BoundingBoxTool.h"
#include "IRenderWindow.h"
#include "Light.h"
#include "MoveTool.h"
#include "RenderSettings.h"
#include "RotateTool.h"
#include "ScaleBarTool.h"
#include "SceneLight.h"
#include "graphics/RenderGL.h"
#include "graphics/RenderGLPT.h"
#include "graphics/GestureGraphicsGL.h"
#include "graphics/gl/Util.h"
#include "renderlib.h"

ViewerWindow::ViewerWindow(RenderSettings* rs)
  : m_renderSettings(rs)
  , m_renderer(new RenderGLPT(rs))
  , m_gestureRenderer(new GestureRendererGL())
  , m_rendererType(1)
{
  gesture.input.reset();

  m_tools.push_back(new ScaleBarTool());
  m_tools.push_back(new AxisHelperTool());
  m_tools.push_back(new BoundingBoxTool());
}

ViewerWindow::~ViewerWindow()
{
  // any tool associated with a selected object should be deleted via the SceneObject, not here.
  select(nullptr);

  if (m_activeTool != &m_defaultTool) {
    ManipulationTool::destroyTool(m_activeTool);
  }
  for (ManipulationTool* tool : m_tools) {
    ManipulationTool::destroyTool(tool);
  }
  m_activeTool = nullptr;
}

void
ViewerWindow::setSize(int width, int height)
{
  sceneView.viewport.region.lower.x = 0;
  sceneView.viewport.region.lower.y = 0;
  sceneView.viewport.region.upper.x = width;
  sceneView.viewport.region.upper.y = height;

  // TODO do whatever resizing now?
}

void
ViewerWindow::select(SceneObject* obj)
{
  // if obj is currently selected then do nothing?
  if (sceneView.getSelectedObject() == obj) {
    return;
  }

  // deselect the current selection
  if (sceneView.getSelectedObject()) {
    ManipulationTool* tool = sceneView.getSelectedObject()->getSelectedTool();
    if (tool) {
      // remove moves tool to end, then erase removes from the new tool position to end
      m_tools.erase(std::remove(m_tools.begin(), m_tools.end(), tool), m_tools.end());
    }
    sceneView.getSelectedObject()->onSelection(false);
  }

  sceneView.setSelectedObject(obj);

  if (obj) {
    obj->onSelection(true);
    ManipulationTool* tool = sceneView.getSelectedObject()->getSelectedTool();
    if (tool) {
      m_tools.push_back(tool);
    }
  }
}

void
ViewerWindow::updateCamera()
{
  // Use gesture strokes (if any) to move the camera. If camera edit is still in progress, we are not
  // going to change the camera directly, instead we fill a CameraModifier object with the delta.
  CameraModifier cameraMod;
  bool cameraEdit = cameraManipulation(glm::vec2(width(), height()),
                                       // m_clock,
                                       gesture,
                                       m_CCamera,
                                       cameraMod);
  // Apply camera animation transitions if we have any
  if (!m_cameraAnim.empty()) {
    for (auto it = m_cameraAnim.begin(); it != m_cameraAnim.end();) {
      CameraAnimation& anim = *it;
      anim.time += m_clock.timeIncrement;

      if (anim.time < anim.duration) { // alpha < 1.0) {
        float alpha = glm::smoothstep(0.0f, 1.0f, glm::clamp(anim.time / anim.duration, 0.0f, 1.0f));
        // Animation in-betweens are accumulated to the camera modifier
        cameraMod = cameraMod + (anim.mod * alpha);
        cameraEdit = true;
        ++it;
      } else {
        // Completed animation is applied to the camera instead
        m_CCamera = m_CCamera + anim.mod;
        it = m_cameraAnim.erase(it);
      }

      // let renderer know camera is dirty
      m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
    }
  }

  // Produce the render camera for current frame
  // m_CCamera.Update();
  CCamera renderCamera = m_CCamera;
  if (cameraEdit) {
    renderCamera = m_CCamera + cameraMod;
  }
  // renderCamera.Update();
  if (cameraEdit) {
    // let renderer know camera is dirty
    m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
  }

  sceneView.camera = renderCamera;
  sceneView.camera.Update();

  // Update lights to maintain fixed direction relative to camera view when enabled.
  // Basic strategy: capture the view space direction of each light at the start of
  // camera manipulation, then during manipulation update the light direction
  // in world space to match the captured relative direction.
  if (sceneView.scene && sceneView.scene->m_lighting.lockToCamera) {
    auto validateBasis = [&](Light* light, const char* label) {
      if (!light) {
        return;
      }
      const float lenN = glm::length(light->m_N);
      const float lenU = glm::length(light->m_U);
      const float lenV = glm::length(light->m_V);
      const float dotNU = glm::dot(light->m_N, light->m_U);
      const float dotNV = glm::dot(light->m_N, light->m_V);
      const float dotUV = glm::dot(light->m_U, light->m_V);
      const float eps = 1e-3f;
      if (glm::abs(lenN - 1.0f) > eps || glm::abs(lenU - 1.0f) > eps || glm::abs(lenV - 1.0f) > eps ||
          glm::abs(dotNU) > eps || glm::abs(dotNV) > eps || glm::abs(dotUV) > eps) {
        LOG_WARNING << "Light basis not orthonormal (" << label << ")"
                    << " lenN=" << lenN << " lenU=" << lenU << " lenV=" << lenV << " dotNU=" << dotNU
                    << " dotNV=" << dotNV << " dotUV=" << dotUV;
      }
    };

    auto captureRelativeBasis = [&](SceneLight* sceneLight, glm::mat3& capturedBasis) {
      if (!sceneLight || !sceneLight->m_light) {
        return;
      }
      CCamera baseCamera = m_CCamera;
      baseCamera.Update();

      glm::vec3 worldU = sceneLight->m_light->m_U;
      glm::vec3 worldV = sceneLight->m_light->m_V;
      glm::vec3 worldN = sceneLight->m_light->m_N;

      glm::vec3 capturedU(
        glm::dot(worldU, baseCamera.m_U), glm::dot(worldU, baseCamera.m_V), glm::dot(worldU, baseCamera.m_N));
      glm::vec3 capturedV(
        glm::dot(worldV, baseCamera.m_U), glm::dot(worldV, baseCamera.m_V), glm::dot(worldV, baseCamera.m_N));
      glm::vec3 capturedN(
        glm::dot(worldN, baseCamera.m_U), glm::dot(worldN, baseCamera.m_V), glm::dot(worldN, baseCamera.m_N));

      capturedBasis = glm::mat3(capturedU, capturedV, capturedN);
    };

    auto applyRelativeBasis = [&](SceneLight* sceneLight, const glm::mat3& capturedBasis) {
      if (!sceneLight || !sceneLight->m_light) {
        return;
      }

      glm::vec3 capturedU = capturedBasis[0];
      glm::vec3 capturedV = capturedBasis[1];
      glm::vec3 capturedN = capturedBasis[2];

      glm::vec3 newU =
        capturedU.x * sceneView.camera.m_U + capturedU.y * sceneView.camera.m_V + capturedU.z * sceneView.camera.m_N;
      glm::vec3 newV =
        capturedV.x * sceneView.camera.m_U + capturedV.y * sceneView.camera.m_V + capturedV.z * sceneView.camera.m_N;
      glm::vec3 newN =
        capturedN.x * sceneView.camera.m_U + capturedN.y * sceneView.camera.m_V + capturedN.z * sceneView.camera.m_N;

      if (glm::length(newN) <= 1e-6f) {
        newN = glm::vec3(0.0f, 0.0f, 1.0f);
      }
      newN = glm::normalize(newN);

      // Orthonormalize basis to preserve roll when possible.
      newU = newU - newN * glm::dot(newU, newN);
      if (glm::length(newU) <= 1e-6f) {
        newU = glm::abs(newN.y) > 0.999f ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
      }
      newU = glm::normalize(newU);
      newV = glm::normalize(glm::cross(newN, newU));

      Light* light = sceneLight->m_light;
      light->m_UseExplicitBasis = true;
      light->m_N = newN;
      light->m_U = newU;
      light->m_V = newV;

      float phi, theta;
      Light::cartesianToSpherical(newN, phi, theta);
      light->m_Phi = phi;
      light->m_Theta = theta;
      light->m_P = light->m_Target + newN;

      light->updateBasisFrame();

      validateBasis(light, "sphere-lock");

      for (auto& observer : sceneLight->m_observers) {
        observer(*light);
      }
    };

    auto applyRelativeBasisAreaLight = [&](SceneLight* sceneLight, const glm::mat3& capturedBasis) {
      if (!sceneLight || !sceneLight->m_light) {
        return;
      }

      glm::vec3 capturedU = capturedBasis[0];
      glm::vec3 capturedV = capturedBasis[1];
      glm::vec3 capturedN = capturedBasis[2];

      glm::vec3 newU =
        capturedU.x * sceneView.camera.m_U + capturedU.y * sceneView.camera.m_V + capturedU.z * sceneView.camera.m_N;
      glm::vec3 newV =
        capturedV.x * sceneView.camera.m_U + capturedV.y * sceneView.camera.m_V + capturedV.z * sceneView.camera.m_N;
      glm::vec3 newN =
        capturedN.x * sceneView.camera.m_U + capturedN.y * sceneView.camera.m_V + capturedN.z * sceneView.camera.m_N;

      if (glm::length(newN) <= 1e-6f) {
        newN = glm::vec3(0.0f, 0.0f, 1.0f);
      }
      newN = glm::normalize(newN);

      newU = newU - newN * glm::dot(newU, newN);
      if (glm::length(newU) <= 1e-6f) {
        newU = glm::abs(newN.y) > 0.999f ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
      }
      newU = glm::normalize(newU);
      newV = glm::normalize(glm::cross(newN, newU));

      Light* light = sceneLight->m_light;
      light->m_U = newU;
      light->m_V = newV;

      glm::vec3 defaultDir(0.0f, 0.0f, 1.0f);
      glm::vec3 newWorldLightDir = -newN;
      glm::quat rotation;
      float dot = glm::dot(defaultDir, newWorldLightDir);
      if (dot < -0.999999f) {
        rotation = glm::angleAxis(glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
      } else {
        glm::vec3 axis = glm::cross(defaultDir, newWorldLightDir);
        rotation = glm::quat(1.0f + dot, axis.x, axis.y, axis.z);
        rotation = glm::normalize(rotation);
      }

      sceneLight->m_transform.m_rotation = rotation;
      sceneLight->updateTransform();

      validateBasis(light, "area-lock");
    };

    if (cameraEdit && !m_wasCameraBeingEdited) {
      captureRelativeBasis(sceneView.scene->SceneAreaLight(), m_capturedAreaLightRelativeBasis);
      captureRelativeBasis(sceneView.scene->SceneSphereLight(), m_capturedSphereLightRelativeBasis);
    }

    if (cameraEdit) {
      applyRelativeBasisAreaLight(sceneView.scene->SceneAreaLight(), m_capturedAreaLightRelativeBasis);
      applyRelativeBasis(sceneView.scene->SceneSphereLight(), m_capturedSphereLightRelativeBasis);
      m_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
    }
  }

  if (sceneView.scene && !sceneView.scene->m_lighting.lockToCamera) {
    SceneLight* sphereLight = sceneView.scene->SceneSphereLight();
    if (sphereLight && sphereLight->m_light) {
      sphereLight->m_light->m_UseExplicitBasis = false;
    }
  }

  // Track camera edit state for next frame
  m_wasCameraBeingEdited = cameraEdit;
}

void
ViewerWindow::update(const SceneView::Viewport& viewport, const Clock& clock, Gesture& gesture)
{
  // [...]

  // TODO FIXME
  // We need a mechanism to add a tool just from a scene object that was added to the scene.
  // For now, we just add the tool for special scene objects(like clip plane) right here on the fly.
  // Instead of adding to this temporary vector, we should add to the sceneView.scene->m_tools
  std::vector<ManipulationTool*> sceneTools;
  if (sceneView.scene) {
    if (sceneView.scene->m_clipPlane) {
      if (sceneView.scene->m_clipPlane->m_enabled) {
        if (sceneView.scene->m_clipPlane->getTool()) {
          // add to sceneTools, a temporary array per-update
          sceneTools.push_back(sceneView.scene->m_clipPlane->getTool());
        }
      }
    }
  }

  // Reset all manipulators and tools
  for (ManipulationTool* tool : sceneTools) {
    tool->clear();
  }
  forEachTool([&](ManipulationTool* tool) { tool->clear(); });

  // Query Gesture::Graphics for selection codes
  // If we are in mid-gesture, then we can continue to use the retained selection code.
  // if we never had anything to draw into the pick buffer, then we didn't pick anything.
  // This is a slight oversimplification because it checks for any single-button release
  // or drag: two-button drag gestures are not handled well.
  bool pickedAnything = false;
  if (gesture.input.clickEnded() || gesture.input.isDragging()) {
    pickedAnything = gesture.graphics.m_retainedSelectionCode != Gesture::Graphics::k_noSelectionCode;
  } else {
    pickedAnything =
      m_gestureRenderer->pick(m_selection, gesture.input, viewport, gesture.graphics.m_retainedSelectionCode);
  }

  if (pickedAnything) {
    int selectionCode = gesture.graphics.getCurrentSelectionCode();
    for (ManipulationTool* tool : sceneTools) {
      tool->setActiveCode(selectionCode);
    }
    forEachTool([&](ManipulationTool* tool) { tool->setActiveCode(selectionCode); });
  } else {
    // User didn't click on a manipulator, run scene object selection
    // [...]
    updateCamera();
  }

  // update sceneView.camera here?
  // It is already being updated inside of updateCamera
  // (only when camera is being manipulated!)

  // Run all manipulators and tools
  {
    // Ask manipulation tools to do something. Typically only one of them does
    // something, if anything at all
    for (ManipulationTool* tool : sceneTools) {
      tool->action(sceneView, gesture);
    }
    forEachTool([&](ManipulationTool* tool) { tool->action(sceneView, gesture); });

    // Leave code 0 as neutral, we can start at any number, this will not affect
    // anything else in the system... start at 1
    int selectionCodeOffset = 1;
    for (ManipulationTool* tool : sceneTools) {
      tool->requestCodesRange(&selectionCodeOffset);
    }
    forEachTool([&](ManipulationTool* tool) { tool->requestCodesRange(&selectionCodeOffset); });

    // Ask tools to generate draw commands
    for (ManipulationTool* tool : sceneTools) {
      tool->draw(sceneView, gesture);
    }
    forEachTool([&](ManipulationTool* tool) { tool->draw(sceneView, gesture); });
  }

  // Manipulators may have changed the scene.
  // Update state, build BVH, etc...
  //[...]
}

void
ViewerWindow::redraw()
{
  m_clock.tick();
  // m_gesture.setTimeIncrement(m_clock.timeIncrement);
  // Display frame rate in window title
  double interval = m_clock.time - m_lastTimeCheck; //< Interval in seconds
  m_increments += 1;
  // update once per second
  if (interval >= 1.0) {
    // Compute average frame rate over the last second, if different than what we
    // display previously, update the window title.
    double newFrameRate = round(m_increments / interval);
    if (m_frameRate != newFrameRate) {
      m_frameRate = newFrameRate;

      // TODO update frame rate stats from here.
      // char title[256];
      // snprintf(title, 256, "%s | %d fps", windowTitle, frameRate);
      // glfwSetWindowTitle(mainWindow.handle, title);
    }
    m_lastTimeCheck = m_clock.time;
    m_increments = 0;
  }

  glm::ivec2 oldpickbuffersize = m_selection.resolution;
  bool ok = m_selection.update(glm::ivec2(width(), height()));
  if (!ok) {
    LOG_ERROR << "Failed to update selection buffer";
  }

  // lazy init
  if (!gesture.graphics.font.isLoaded()) {
    std::string fontPath = renderlib::assetPath() + "/fonts/Arial.ttf";
    gesture.graphics.font.load(fontPath.c_str());
  }

  // renderer size may have been directly manipulated by e.g. the renderdialog
  uint32_t oldrendererwidth, oldrendererheight;
  m_renderer->getSize(oldrendererwidth, oldrendererheight);
  if (width() != oldpickbuffersize.x || height() != oldpickbuffersize.y || width() != oldrendererwidth ||
      height() != oldrendererheight) {
    m_renderer->resize(width(), height());
    m_CCamera.m_Film.m_Resolution.SetResX(width());
    m_CCamera.m_Film.m_Resolution.SetResY(height());
  }

  sceneView.viewport.region = { { 0, 0 }, { width(), height() } };
  sceneView.camera = m_CCamera;
  sceneView.scene = m_renderer->scene();
  sceneView.renderSettings = m_renderSettings;

  // fill gesture graphics with draw commands
  update(sceneView.viewport, m_clock, gesture);

  // ready to start drawing; clear our main framebuffer
  clearFramebuffer(sceneView.scene);

  // render and then clear out draw commands from gesture graphics
  m_gestureRenderer->drawUnderlay(sceneView, &m_selection, gesture.graphics);

  // main scene rendering; need to blend/composite on top of overlay previously drawn
  m_renderer->render(sceneView.camera);

  // render and then clear out draw commands from gesture graphics
  m_gestureRenderer->draw(sceneView, &m_selection, gesture.graphics);

  // Make sure we consumed any unused input event before we poll new events.
  // (in the case of Qt we are not explicitly polling but using signals/slots.)
  gesture.input.consume();
}

void
ViewerWindow::setRenderer(int rendererType)
{
  // clean up old renderer.
  if (m_renderer) {
    m_renderer->cleanUpResources();
  }

  Scene* sc = m_renderer->scene();

  switch (rendererType) {
    case 1:
      LOG_DEBUG << "Set OpenGL pathtrace Renderer";
      m_renderer.reset(new RenderGLPT(m_renderSettings));
      m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
      break;
    case 2:
      LOG_DEBUG << "Set OpenGL pathtrace Renderer";
      m_renderer.reset(new RenderGLPT(m_renderSettings));
      m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
      break;
    default:
      LOG_DEBUG << "Set OpenGL single pass Renderer";
      m_renderer.reset(new RenderGL(m_renderSettings));
  };
  m_rendererType = rendererType;

  // need to update the scene in QAppearanceSettingsWidget.
  m_renderer->setScene(sc);
  m_renderer->initialize(width(), height()); // TODO , devicePixelRatioF());

  m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
