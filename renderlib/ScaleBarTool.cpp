#include "ScaleBarTool.h"

#include "ImageXYZC.h"
#include "Logging.h"

#include <iomanip>
#include <sstream>

float
getScaleBarWidth(int windowWidthPx, float windowPhysicalWidth)
{
  // We want to find the largest round number of physical units that keeps the scale bar within this width on screen
  const int SCALE_BAR_MAX_WIDTH = (int)((float)windowWidthPx * 0.333f);
  // Convert max width to volume physical units
  const float physicalMaxWidth = (float)SCALE_BAR_MAX_WIDTH * windowPhysicalWidth / (float)windowWidthPx;
  // Round off all but the most significant digit of physicalMaxWidth
  const float digits = floor(log10(physicalMaxWidth));
  const float div10 = pow(10.0, digits);
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
  if (scene.camera.m_Projection != ProjectionMode::ORTHOGRAPHIC) {
    return;
  }

  glm::vec3 color = glm::vec3(1, 1, 1);
  float opacity = 1.0f;
  uint32_t code = Gesture::Graphics::SelectionBuffer::k_noSelectionCode;
  gesture.graphics.addCommand(GL_LINES, Gesture::Graphics::CommandSequence::k2dScreen);

  // get physical size of volume
  auto volume = scene.scene->m_volume;
  const glm::vec3 volumePhysicalSize(volume->physicalSizeX() * (float)volume->sizeX(),
                                     volume->physicalSizeY() * (float)volume->sizeY(),
                                     volume->physicalSizeZ() * (float)volume->sizeZ());

  // the max value is the one that corresponds to the window height.
  // (see ImageXYZC::getDimensions() for the math)
  float orthoHeight = std::max(volumePhysicalSize.x, std::max(volumePhysicalSize.y, volumePhysicalSize.z));
  orthoHeight *= (scene.camera.m_OrthoScale * 2.0f);
  const float aspect = (float)scene.viewport.region.size().x / (float)scene.viewport.region.size().y;
  float orthoWidth = orthoHeight * aspect;

  const float scaleBarWidthPx = getScaleBarWidth(scene.viewport.region.size().x, orthoWidth);
  // now get a scale bar width that is 1/10 the width of the window:
  // const float scaleBarWindowFraction = 0.1;
  // const float scaleBarWidth = orthoWidth * scaleBarWindowFraction;
  // static float lastScaleWidth = scaleBarWidthPx;
  // if (scaleBarWidthPx != lastScaleWidth) {
  //   LOG_DEBUG << "scaleBarWidth changed from " << lastScaleWidth << " to " << scaleBarWidthPx;
  //   lastScaleWidth = scaleBarWidthPx;
  // }

  // 0,0 is lower left, size() is upper right.  position bar at lower right.
  glm::vec3 p0 = glm::vec3(scene.viewport.region.size().x, 0, 1.0f) + glm::vec3(-40, 40, 0);
  glm::vec3 p1 = p0 - glm::vec3(scaleBarWidthPx, 0, 0);
  // draw one horizontal line about 1/10 the width of the window, with short bars at the ends.
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
                           Gesture::Graphics::VertsCode(p1, color, opacity, code));

  glm::vec3 v = glm::vec3(0, 15, 0);
  // draw lines from the middle of the light to the target
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p0 + v, color, opacity, code),
                           Gesture::Graphics::VertsCode(p0 - v, color, opacity, code));
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p1 + v, color, opacity, code),
                           Gesture::Graphics::VertsCode(p1 - v, color, opacity, code));

  // draw text
  // TODO utilize m_size to scale the font quads
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2)
         << (scaleBarWidthPx * orthoWidth / (float)scene.viewport.region.size().x);
  stream << " " << scene.scene->m_volume->spatialUnits();
  std::string msg = stream.str();
  float wid = gesture.graphics.font->getStringWidth(msg);
  gesture.graphics.addCommand(GL_TRIANGLES, Gesture::Graphics::CommandSequence::k2dScreen);
  gesture.drawText(msg, p1 + glm::vec3(scaleBarWidthPx * 0.5 - wid * 0.5, 20, 0), color, opacity, code);
}
