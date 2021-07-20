#include "Status.h"

#include "AppScene.h"
#include "ImageXYZC.h"

// Render status singleton
CStatus gStatus;

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
  for (IStatusObserver* ob : mObservers) {
    ob->RenderBegin();
  }
}

void
CStatus::SetRenderEnd(void)
{
  for (IStatusObserver* ob : mObservers) {
    ob->RenderEnd();
  }
}

void
CStatus::SetPreRenderFrame(void)
{
  for (IStatusObserver* ob : mObservers) {
    ob->PreRenderFrame();
  }
}

void
CStatus::SetPostRenderFrame(void)
{
  for (IStatusObserver* ob : mObservers) {
    ob->PostRenderFrame();
  }
}

void
CStatus::SetRenderPause(const bool& Pause)
{
  for (IStatusObserver* ob : mObservers) {
    ob->RenderPause(Pause);
  }
}

void
CStatus::SetResize(void)
{
  for (IStatusObserver* ob : mObservers) {
    ob->Resize();
  }
}

void
CStatus::SetLoadPreset(const std::string& PresetName)
{
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
  for (IStatusObserver* ob : mObservers) {
    ob->StatisticChanged(Group, Name, Value, Unit, Icon);
  }
}

void
CStatus::onNewImage(const std::string& name, Scene* scene)
{
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
  // s->SetStatisticChanged("Volume", "No. Voxels", QString::number((double)resolution.x * (double)resolution.y *
  // (double)resolution.z), "Voxels");
  // TODO: this is per channel
  // s->SetStatisticChanged("Volume", "Density Range", "[" + QString::number(gScene.m_IntensityRange.GetMin()) + ", " +
  // QString::number(gScene.m_IntensityRange.GetMax()) + "]", "");
}
