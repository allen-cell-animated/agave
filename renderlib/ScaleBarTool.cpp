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
  // const float digits = floor(log10(physicalMaxWidth));
  // const float div10 = pow(10.0, digits);
  const float scaleValue = floor(physicalMaxWidth / (float)div10) * div10;
  static float lastScaleValue = scaleValue;
  if (scaleValue != lastScaleValue) {
    LOG_DEBUG << "scaleValue changed from " << lastScaleValue << " to " << scaleValue
              << " (px = " << (SCALE_BAR_MAX_WIDTH * (scaleValue / physicalMaxWidth)) << ")";
    lastScaleValue = scaleValue;
  }

  // convert to string
  // let scaleStr = scaleValue.toString();
  // if (digits < 1) {
  //  // Handle irrational floating point values (e.g. 0.30000000000000004)
  //  scaleStr = scaleStr.slice(0, Math.abs(digits) + 2);
  //}
  // this.orthoScaleBarElement.innerHTML = `${scaleStr}${unit || ""}`;
  // this.orthoScaleBarElement.style.width = `${SCALE_BAR_MAX_WIDTH * (scaleValue / physicalMaxWidth)}px`;
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
  const float SCALE_BAR_MAX_WIDTH = (windowWidthPx * 0.25f);
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
    return { physicalSize, SCALE_BAR_MAX_WIDTH };
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

  ScaleBarSize scaleBarSize = computeScaleBarSize(scene);

  glm::vec3 color = glm::vec3(1, 1, 1);
  float opacity = 1.0f;
  uint32_t code = Gesture::Graphics::SelectionBuffer::k_noSelectionCode;
  gesture.graphics.addCommand(GL_LINES, Gesture::Graphics::CommandSequence::k2dScreen);

  static const glm::vec3 offsetFromBottomRight(-40.0f, 40.0f, 0.0f);
  // 0,0 is lower left, size() is upper right.  position bar at lower right.
  glm::vec3 p0 = glm::vec3(scene.viewport.region.size().x, 0, 1.0f) + offsetFromBottomRight;
  glm::vec3 p1 = p0 - glm::vec3(scaleBarSize.pixels, 0, 0);
  // draw one horizontal line about 1/10 the width of the window, with short bars at the ends.
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
                           Gesture::Graphics::VertsCode(p1, color, opacity, code));

  glm::vec3 v = glm::vec3(0, 15, 0);
  // draw tick lines at the ends
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p0 + v, color, opacity, code),
                           Gesture::Graphics::VertsCode(p0 - v, color, opacity, code));
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p1 + v, color, opacity, code),
                           Gesture::Graphics::VertsCode(p1 - v, color, opacity, code));

  // draw text
  // TODO utilize m_size to scale the font quads
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << scaleBarSize.physicalSize;
  stream << " " << scene.scene->m_volume->spatialUnits();
  std::string msg = stream.str();
  float wid = gesture.graphics.font->getStringWidth(msg);
  gesture.graphics.addCommand(GL_TRIANGLES, Gesture::Graphics::CommandSequence::k2dScreen);
  gesture.drawText(msg, p1 + glm::vec3(scaleBarSize.pixels * 0.5 - wid * 0.5, 20, 0), color, opacity, code);
}
