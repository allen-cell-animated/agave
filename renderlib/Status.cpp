#include "Stable.h"

#include "Status.h"

#include "AppScene.h"
#include "ImageXYZC.h"

// Render status singleton
CStatus gStatus;

inline QString FormatVector(const glm::vec3& Vector, const int& Precision = 2)
{
	return "[" + QString::number(Vector.x, 'f', Precision) + ", " + QString::number(Vector.y, 'f', Precision) + ", " + QString::number(Vector.z, 'f', Precision) + "]";
}

inline QString FormatVector(const glm::ivec3& Vector)
{
	return "[" + QString::number(Vector.x) + ", " + QString::number(Vector.y) + ", " + QString::number(Vector.z) + "]";
}

inline QString FormatSize(const glm::vec3& Size, const int& Precision = 2)
{
	return QString::number(Size.x, 'f', Precision) + " x " + QString::number(Size.y, 'f', Precision) + " x " + QString::number(Size.z, 'f', Precision);
}

inline QString FormatSize(const glm::ivec3& Size)
{
	return QString::number(Size.x) + " x " + QString::number(Size.y) + " x " + QString::number(Size.z);
}

void CStatus::SetRenderBegin(void)
{
	emit RenderBegin();
}

void CStatus::SetRenderEnd(void)
{
	emit RenderEnd();
}

void CStatus::SetPreRenderFrame(void)
{
	emit PreRenderFrame();
}

void CStatus::SetPostRenderFrame(void)
{
	emit PostRenderFrame();
}

void CStatus::SetRenderPause(const bool& Pause)
{
	emit RenderPause(Pause);
}

void CStatus::SetResize(void)
{
	emit Resize();
}

void CStatus::SetLoadPreset(const QString& PresetName)
{
	emit LoadPreset(PresetName);
}

void CStatus::SetStatisticChanged(const QString& Group, const QString& Name, const QString& Value, const QString& Unit /*= ""*/, const QString& Icon /*= ""*/)
{
	emit StatisticChanged(Group, Name, Value, Unit, Icon);
}

void CStatus::onNewImage(const QString& name, Scene* scene)
{
	glm::vec3 resolution(scene->_volume->sizeX(), scene->_volume->sizeY(), scene->_volume->sizeZ());
	glm::vec3 spacing(scene->_volume->physicalSizeX(), scene->_volume->physicalSizeY(), scene->_volume->physicalSizeZ());
	const glm::vec3 PhysicalSize(
		spacing.x * (float)resolution.x,
		spacing.y * (float)resolution.y,
		spacing.z * (float)resolution.z
	);
	glm::vec3 BoundingBoxMinP = glm::vec3(0.0f);
	glm::vec3 BoundingBoxMaxP = PhysicalSize / std::max(PhysicalSize.x, std::max(PhysicalSize.y, PhysicalSize.z));
	
	SetStatisticChanged("Volume", "File", name, "");
	SetStatisticChanged("Volume", "Bounding Box", "", "");
	SetStatisticChanged("Bounding Box", "Min", FormatVector(BoundingBoxMinP, 2), "");
	SetStatisticChanged("Bounding Box", "Max", FormatVector(BoundingBoxMaxP, 2), "");
	SetStatisticChanged("Volume", "Physical Size", FormatSize(PhysicalSize, 2), "");
	SetStatisticChanged("Volume", "Resolution", FormatSize(resolution), "");
	SetStatisticChanged("Volume", "Spacing", FormatSize(spacing, 2), "");
	//s->SetStatisticChanged("Volume", "No. Voxels", QString::number((double)resolution.x * (double)resolution.y * (double)resolution.z), "Voxels");
	// TODO: this is per channel
	//s->SetStatisticChanged("Volume", "Density Range", "[" + QString::number(gScene.m_IntensityRange.GetMin()) + ", " + QString::number(gScene.m_IntensityRange.GetMax()) + "]", "");
}
