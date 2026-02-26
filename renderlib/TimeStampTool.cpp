#include "TimeStampTool.h"

#include "AppScene.h"
#include "Timeline.h"

#include <cmath>
#include <iomanip>
#include <sstream>

static std::string
formatTimeHhMmSs(double seconds)
{
  if (seconds < 0.0) {
    seconds = 0.0;
  }
  int64_t totalSeconds = static_cast<int64_t>(std::floor(seconds));
  int64_t hours = totalSeconds / 3600;
  int64_t minutes = (totalSeconds % 3600) / 60;
  int64_t secs = totalSeconds % 60;

  std::ostringstream stream;
  stream << std::setfill('0') << std::setw(2) << hours << ":" << std::setw(2) << minutes << ":" << std::setw(2) << secs;
  return stream.str();
}

void
TimeStampTool::action(SceneView& scene, Gesture& gesture)
{
}

void
TimeStampTool::draw(SceneView& scene, Gesture& gesture)
{
  if (!scene.scene) {
    return;
  }
  if (!scene.scene->m_showTimeStamp) {
    return;
  }

  const float windowWidthPx = (float)scene.viewport.region.size().x;
  const float windowHeightPx = (float)scene.viewport.region.size().y;
  static constexpr float WINDOW_REFERENCE_RES = 1200.0f;
  glm::vec3 pctToPx(windowWidthPx / WINDOW_REFERENCE_RES, windowHeightPx / WINDOW_REFERENCE_RES, 1.0f);

  const double physicalSeconds = scene.scene->m_timeLine.currentPhysicalTime();
  const std::string msg = formatTimeHhMmSs(physicalSeconds);

  glm::vec3 color = glm::vec3(1, 1, 1);
  float opacity = 1.0f;
  uint32_t code = Gesture::Graphics::k_noSelectionCode;

  glm::vec2 textScale = glm::vec2(pctToPx.x, pctToPx.y);
  float textWidth = gesture.graphics.font.getStringWidth(msg) * textScale.x;
  float textHeight = gesture.graphics.font.getStringHeight(msg) * textScale.y;

  gesture.graphics.addCommand(Gesture::Graphics::PrimitiveType::kTriangles,
                              Gesture::Graphics::CommandSequence::k2dScreen);

  const glm::vec3 offsetFromTopRight = glm::vec3(-40.0f, -40.0f, 0.0f) * pctToPx;
  glm::vec3 anchor =
    glm::vec3(scene.viewport.region.size().x, scene.viewport.region.size().y, 1.0f) + offsetFromTopRight;
  glm::vec3 textPos = anchor + glm::vec3(-textWidth, -textHeight, 0.0f);

  gesture.drawText(msg, textPos, textScale, color, opacity, code);
}
