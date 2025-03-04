#include "ViewerWindow.h"

#include "../renderlib_wgpu/RenderWgpuPT.h"
#include "AreaLightTool.h"
#include "AxisHelperTool.h"
#include "IRenderWindow.h"
#include "MoveTool.h"
#include "RenderSettings.h"
#include "RotateTool.h"
#include "ScaleBarTool.h"
#include "graphics/RenderGL.h"
#include "graphics/RenderGLPT.h"
#include "graphics/GestureGraphicsGL.h"
#include "renderlib.h"

ViewerWindow::ViewerWindow(RenderSettings* rs)
  : m_renderSettings(rs)
  , m_renderer(new RenderWgpuPT(rs))
  , m_gestureRenderer(new GestureRendererGL())
  , m_rendererType(1)
{
  gesture.input.reset();

  m_tools.push_back(new ScaleBarTool());
  m_tools.push_back(new AxisHelperTool());
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
  }

  sceneView.setSelectedObject(obj);

  if (obj) {
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
}

void
ViewerWindow::update(const SceneView::Viewport& viewport, const Clock& clock, Gesture& gesture)
{
  // [...]

  // Reset all manipulators and tools
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
    forEachTool([&](ManipulationTool* tool) { tool->action(sceneView, gesture); });

    // Leave code 0 as neutral, we can start at any number, this will not affect
    // anything else in the system... start at 1
    int selectionCodeOffset = 1;
    forEachTool([&](ManipulationTool* tool) { tool->requestCodesRange(&selectionCodeOffset); });

    // Ask tools to generate draw commands
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

  // main scene rendering
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
      m_renderer.reset(new RenderWgpuPT(m_renderSettings));
      m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
      break;
    case 2:
      LOG_DEBUG << "Set OpenGL pathtrace Renderer";
      m_renderer.reset(new RenderWgpuPT(m_renderSettings));
      m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
      break;
    default:
      LOG_DEBUG << "Set OpenGL single pass Renderer";
      m_renderer.reset(new RenderWgpuPT(m_renderSettings));
  };
  m_rendererType = rendererType;

  // need to update the scene in QAppearanceSettingsWidget.
  m_renderer->setScene(sc);
  m_renderer->initialize(width(), height()); // TODO , devicePixelRatioF());

  m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
