#include "ScaleBarTool.h"

#include "ImageXYZC.h"
#include "Logging.h"

#include <iomanip>
#include <sstream>

// physicalscale is max of physical dims x,y,z
static float
computePhysicalScaleBarSize(const float physicalScale)
{
  // note this result will always be some integer power of 10 independent of zoom...
  return pow(10.0f, floor(log10(physicalScale / 2.0f)));
}

static float
getScaleBarWidthOrtho(int windowWidthPx, float windowPhysicalWidth)
{
  // We want to find the largest round number of physical units that keeps the scale bar within this width on screen
  const int SCALE_BAR_MAX_WIDTH = (int)((float)windowWidthPx * 0.25f);
  // Convert max width to volume physical units
  const float physicalMaxWidth = (float)SCALE_BAR_MAX_WIDTH * windowPhysicalWidth / (float)windowWidthPx;
  // Round off all but the most significant digit of physicalMaxWidth
  const float div10 = computePhysicalScaleBarSize(physicalMaxWidth);
  const float scaleValue = floor(physicalMaxWidth / (float)div10) * div10;
  static float lastScaleValue = scaleValue;
  if (scaleValue != lastScaleValue) {
    LOG_DEBUG << "scaleValue changed from " << lastScaleValue << " to " << scaleValue
              << " (px = " << (SCALE_BAR_MAX_WIDTH * (scaleValue / physicalMaxWidth)) << ")";
    lastScaleValue = scaleValue;
  }

  // return scaleValue in pixels
  return (float)SCALE_BAR_MAX_WIDTH * (scaleValue / physicalMaxWidth);
}

struct ScaleBarSize
{
  float physicalSize;
  float pixels;
};

static ScaleBarSize
computeScaleBarSize(SceneView& scene)
{
  const bool isOrtho = (scene.camera.m_Projection == ProjectionMode::ORTHOGRAPHIC);
  const float windowWidthPx = (float)scene.viewport.region.size().x;
  const float windowHeightPx = (float)scene.viewport.region.size().y;
  // get physical size of volume
  auto volume = scene.scene->m_volume;
  const glm::vec3 volumePhysicalSize = volume->getPhysicalDimensions();
  float physicalMaxDim = std::max(volumePhysicalSize.x, std::max(volumePhysicalSize.y, volumePhysicalSize.z));
  if (isOrtho) {
    // the physical max dim value is the one that corresponds to the window height.
    // (see ImageXYZC::getNormalizedDimensions() for the math)
    float orthoHeight = physicalMaxDim * (scene.camera.m_OrthoScale * 2.0f);
    const float aspect = windowWidthPx / windowHeightPx;
    float orthoWidth = orthoHeight * aspect;
    const float scaleBarWidthPx = getScaleBarWidthOrtho(windowWidthPx, orthoWidth);
    return { (scaleBarWidthPx * orthoWidth / windowWidthPx), scaleBarWidthPx };
  } else {
    // Round off all but the most significant digit of physicalMaxDim
    const float physicalSize = computePhysicalScaleBarSize(physicalMaxDim);
    return { physicalSize, windowWidthPx * 0.2f };
  }
}

void
ScaleBarTool::action(SceneView& scene, Gesture& gesture)
{
}

void
ScaleBarTool::draw(SceneView& scene, Gesture& gesture)
{
  if (!scene.scene) {
    return;
  }
  if (!scene.scene->m_showScaleBar) {
    return;
  }

  const float windowWidthPx = (float)scene.viewport.region.size().x;
  const float windowHeightPx = (float)scene.viewport.region.size().y;
  // this number is chosen so that scalebar looks sized correctly in a 1200x1200 window
  // with the default font size
  static constexpr float WINDOW_REFERENCE_RES = 1200.0f;
  glm::vec3 pctToPx(windowWidthPx / WINDOW_REFERENCE_RES, windowHeightPx / WINDOW_REFERENCE_RES, 1.0f);

  ScaleBarSize scaleBarSize = computeScaleBarSize(scene);

  glm::vec3 color = glm::vec3(1, 1, 1);
  float opacity = 1.0f;
  uint32_t code = Gesture::Graphics::SelectionBuffer::k_noSelectionCode;
  gesture.graphics.addCommand(GL_LINES, Gesture::Graphics::CommandSequence::k2dScreen);

  // 0,0 is lower left, size() is upper right.  position bar at lower right.
  const glm::vec3 offsetFromBottomRight = glm::vec3(-40.0f, 40.0f, 0.0f) * pctToPx;
  // p0 is the rightmost point, p1 is the leftmost point
  glm::vec3 p0 = glm::vec3(scene.viewport.region.size().x, 0, 1.0f) + offsetFromBottomRight;
  glm::vec3 p1 = p0 - glm::vec3(scaleBarSize.pixels, 0, 0);
  // draw one horizontal line about 1/10 the width of the window, with short bars at the ends.
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
                           Gesture::Graphics::VertsCode(p1, color, opacity, code));

  glm::vec3 v = glm::vec3(0, 15.0f, 0) * pctToPx;
  // draw tick lines at the ends
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p0 + v, color, opacity, code),
                           Gesture::Graphics::VertsCode(p0 - v, color, opacity, code));
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p1 + v, color, opacity, code),
                           Gesture::Graphics::VertsCode(p1 - v, color, opacity, code));

  // draw text
  glm::vec2 textScale = glm::vec2(pctToPx.x, pctToPx.y);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << scaleBarSize.physicalSize;
  stream << " " << scene.scene->m_volume->spatialUnits();
  std::string msg = stream.str();
  float textWidth = gesture.graphics.font->getStringWidth(msg) * textScale.x;
  float textHeight = gesture.graphics.font->getStringHeight(msg) * textScale.y;
  gesture.graphics.addCommand(GL_TRIANGLES, Gesture::Graphics::CommandSequence::k2dScreen);
  glm::vec3 textoffset = glm::vec3(scaleBarSize.pixels * 0.5 - textWidth * 0.5, textHeight, 0);
  gesture.drawText(msg, p1 + textoffset, textScale, color, opacity, code);
}
