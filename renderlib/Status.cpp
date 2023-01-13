#include "Status.h"

#include "AppScene.h"
#include "ImageXYZC.h"

inline std::string
FormatVector(const glm::vec3& Vector, const int& Precision = 2)
{
  std::ostringstream ss;
  ss << "[" << Vector.x << ", " << Vector.y << ", " << Vector.z << "]";
  return ss.str();
}

inline std::string
FormatVector(const glm::ivec3& Vector)
{
  std::ostringstream ss;
  ss << "[" << Vector.x << ", " << Vector.y << ", " << Vector.z << "]";
  return ss.str();
}

inline std::string
FormatSize(const glm::vec3& Size, const int& Precision = 2)
{
  std::ostringstream ss;
  ss << Size.x << " x " << Size.y << " x " << Size.z;
  return ss.str();
}

inline std::string
FormatSize(const glm::ivec3& Size)
{
  std::ostringstream ss;
  ss << Size.x << " x " << Size.y << " x " << Size.z;
  return ss.str();
}

void
CStatus::SetRenderBegin(void)
{
  if (!mUpdatesEnabled) {
    return;
  }
  for (IStatusObserver* ob : mObservers) {
    ob->RenderBegin();
  }
}

void
CStatus::SetRenderEnd(void)
{
  if (!mUpdatesEnabled) {
    return;
  }
  for (IStatusObserver* ob : mObservers) {
    ob->RenderEnd();
  }
}

void
CStatus::SetPreRenderFrame(void)
{
  if (!mUpdatesEnabled) {
    return;
  }
  for (IStatusObserver* ob : mObservers) {
    ob->PreRenderFrame();
  }
}

void
CStatus::SetPostRenderFrame(void)
{
  if (!mUpdatesEnabled) {
    return;
  }
  for (IStatusObserver* ob : mObservers) {
    ob->PostRenderFrame();
  }
}

void
CStatus::SetRenderPause(const bool& Pause)
{
  if (!mUpdatesEnabled) {
    return;
  }
  for (IStatusObserver* ob : mObservers) {
    ob->RenderPause(Pause);
  }
}

void
CStatus::SetResize(void)
{
  if (!mUpdatesEnabled) {
    return;
  }
  for (IStatusObserver* ob : mObservers) {
    ob->Resize();
  }
}

void
CStatus::SetLoadPreset(const std::string& PresetName)
{
  if (!mUpdatesEnabled) {
    return;
  }
  for (IStatusObserver* ob : mObservers) {
    ob->LoadPreset(PresetName);
  }
}

void
CStatus::SetStatisticChanged(const std::string& Group,
                             const std::string& Name,
                             const std::string& Value,
                             const std::string& Unit /*= ""*/,
                             const std::string& Icon /*= ""*/)
{
  if (!mUpdatesEnabled) {
    return;
  }
  for (IStatusObserver* ob : mObservers) {
    ob->StatisticChanged(Group, Name, Value, Unit, Icon);
  }
}

void
CStatus::onNewImage(const std::string& name, Scene* scene)
{
  if (!mUpdatesEnabled) {
    return;
  }

  glm::vec3 resolution(scene->m_volume->sizeX(), scene->m_volume->sizeY(), scene->m_volume->sizeZ());
  glm::vec3 spacing(
    scene->m_volume->physicalSizeX(), scene->m_volume->physicalSizeY(), scene->m_volume->physicalSizeZ());
  const glm::vec3 PhysicalSize(
    spacing.x * (float)resolution.x, spacing.y * (float)resolution.y, spacing.z * (float)resolution.z);
  glm::vec3 BoundingBoxMinP = glm::vec3(0.0f);
  glm::vec3 BoundingBoxMaxP = PhysicalSize / std::max(PhysicalSize.x, std::max(PhysicalSize.y, PhysicalSize.z));

  SetStatisticChanged("Volume", "File", name, "");
  SetStatisticChanged("Volume", "Bounding Box", "", "");
  SetStatisticChanged("Bounding Box", "Min", FormatVector(BoundingBoxMinP, 2), "");
  SetStatisticChanged("Bounding Box", "Max", FormatVector(BoundingBoxMaxP, 2), "");
  SetStatisticChanged("Volume", "Physical Size", FormatSize(PhysicalSize, 2), "");
  SetStatisticChanged("Volume", "Resolution", FormatSize(resolution), "");
  SetStatisticChanged("Volume", "Spacing", FormatSize(spacing, 2), "");
}
