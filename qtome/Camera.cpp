#include "Stable.h"

#include "Camera.h"

#include "renderlib/RenderSettings.h"

QCamera::QCamera(QObject* pParent /*= NULL*/) :
	QObject(pParent),
	m_Film(),
	m_Aperture(),
	m_Projection(),
	m_Focus(),
	m_From(1.0f),
	m_Target(0.5f),
	m_Up(0.0f, 1.0f, 0.0f)
{
}

QCamera::~QCamera(void)
{
}

QCamera::QCamera(const QCamera& Other)
{
	*this = Other;
};

QCamera& QCamera::operator=(const QCamera& Other)
{
	blockSignals(true);

	m_Film			= Other.m_Film;
	m_Aperture		= Other.m_Aperture;
	m_Projection	= Other.m_Projection;
	m_Focus			= Other.m_Focus;
	m_From			= Other.m_From;
	m_Target		= Other.m_Target;
	m_Up			= Other.m_Up;

	blockSignals(false);

	emit Changed();

	return *this;
}

QFilm& QCamera::GetFilm(void)
{
	return m_Film;
}

void QCamera::SetFilm(const QFilm& Film)
{
	m_Film = Film;
}

QAperture& QCamera::GetAperture(void)
{
	return m_Aperture;
}

void QCamera::SetAperture(const QAperture& Aperture)
{
	m_Aperture = Aperture;
}

QProjection& QCamera::GetProjection(void)
{
	return m_Projection;
}

void QCamera::SetProjection(const QProjection& Projection)
{
	m_Projection = Projection;
}

QFocus& QCamera::GetFocus(void)
{
	return m_Focus;
}

void QCamera::SetFocus(const QFocus& Focus)
{
	m_Focus = Focus;
}

Vec3f QCamera::GetFrom(void) const
{
	return m_From;
}

void QCamera::SetFrom(const Vec3f& From)
{
	m_From = From;

	emit Changed();
}

Vec3f QCamera::GetTarget(void) const
{
	return m_Target;
}

void QCamera::SetTarget(const Vec3f& Target)
{
	m_Target = Target;

	emit Changed();
}

Vec3f QCamera::GetUp(void) const
{
	return m_Up;
}

void QCamera::SetUp(const Vec3f& Up)
{
	m_Up = Up;

	emit Changed();
}

QCamera QCamera::Default(void)
{
	QCamera DefaultCamera;

	//DefaultCamera.SetName("Default");

	return DefaultCamera;
}

void QCamera::OnFilmChanged(void)
{
	emit Changed();
}

void QCamera::OnApertureChanged(void)
{
	emit Changed();
}

void QCamera::OnProjectionChanged(void)
{
 	emit Changed();
}

void QCamera::OnFocusChanged(void)
{
	emit Changed();
}

