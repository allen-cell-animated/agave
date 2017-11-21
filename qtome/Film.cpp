#include "Stable.h"

#include "Film.h"

QFilm::QFilm(QObject* pParent /*= NULL*/) :
	QObject(pParent),
	m_Width(800),
	m_Height(600),
	m_Exposure(0.75f),
	m_ExposureIterations(1),
	m_Dirty(false)
{
}

QFilm::QFilm(const QFilm& Other)
{
	*this = Other;
}

QFilm& QFilm::operator=(const QFilm& Other)
{
	m_Width				= Other.m_Width;
	m_Height			= Other.m_Height;
	m_Exposure = Other.m_Exposure;
	m_ExposureIterations = Other.m_ExposureIterations;
	m_NoiseReduction	= Other.m_NoiseReduction;
	m_Dirty				= Other.m_Dirty;

	emit Changed(*this);

	return *this;
}

int QFilm::GetWidth(void) const
{
	return m_Width;
}

void QFilm::SetWidth(const int& Width)
{
	m_Width		= Width;
	m_Dirty		= true;

	emit Changed(*this);
}

int QFilm::GetHeight(void) const
{
	return m_Height;
}

void QFilm::SetHeight(const int& Height)
{
	m_Height	= Height;
	m_Dirty		= true;

	emit Changed(*this);
}

float QFilm::GetExposure(void) const
{
	return m_Exposure;
}

void QFilm::SetExposure(const float& Exposure)
{
	m_Exposure = Exposure;
	emit Changed(*this);
}

int QFilm::GetExposureIterations(void) const
{
	return m_ExposureIterations;
}

void QFilm::SetExposureIterations(const int& ExposureIterations)
{
	m_ExposureIterations = ExposureIterations;
	emit Changed(*this);
}

bool QFilm::GetNoiseReduction(void) const
{
	return m_NoiseReduction;
}

void QFilm::SetNoiseReduction(const bool& NoiseReduction)
{
	m_NoiseReduction = NoiseReduction;
	emit Changed(*this);
}

bool QFilm::IsDirty(void) const
{
	return m_Dirty;
}

void QFilm::UnDirty(void)
{
	m_Dirty = false;
}

