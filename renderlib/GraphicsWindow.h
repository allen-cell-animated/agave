#pragma once

#include "CCamera.h"
#include "Manipulator.h"
#include "Timing.h"
#include "gesture/gesture.h"

#include <vector>

class RenderSettings;
class IRenderWindow;

class ViewerWindow
{
public:
  ViewerWindow(RenderSettings* rs);
  ~ViewerWindow();

  void setSize(int width, int height);
  int width() const { return sceneView.viewport.region.upper.x - sceneView.viewport.region.lower.x; }
  int height() const { return sceneView.viewport.region.upper.y - sceneView.viewport.region.lower.y; }

  void redraw();

  void update(const SceneView::Viewport& viewport, const Clock& clock, Gesture& gesture);

  void setRenderer(int rendererType);

  // Provide a new active tool
  void setTool(ManipulationTool* tool)
  {
    if (m_activeTool != &m_defaultTool) {
      ManipulationTool::destroyTool(m_activeTool);
    }
    m_activeTool = (tool ? tool : &m_defaultTool);

    // Todo: this could be replaced with a push/pop mechanism to allow
    //       the completion of a tool to restore a previous state.
  }

  // A convenient template helper function to run all the instantiated
  // manipulator tools trough some computation (an arbitrary lambda function)
  template<typename Fn>
  void forEachTool(Fn fn)
  {
    for (ManipulationTool* tool : m_tools)
      fn(tool);
    fn(m_activeTool);
  }

  void updateCamera();

  CCamera m_CCamera;
  std::vector<CameraAnimation> m_cameraAnim;

  Gesture gesture;
  SceneView sceneView;
  Clock m_clock;
  double m_lastTimeCheck;
  double m_frameRate;
  int m_increments;

  Gesture::Graphics::SelectionBuffer m_selection;

  ManipulationTool m_defaultTool; //< a null tool representing selection
  ManipulationTool* m_activeTool = &m_defaultTool;
  std::vector<ManipulationTool*> m_tools;
  bool m_toolsUseLocalSpace = false;

  RenderSettings* m_renderSettings;
  std::unique_ptr<IRenderWindow> m_renderer;
  int m_rendererType;
};
