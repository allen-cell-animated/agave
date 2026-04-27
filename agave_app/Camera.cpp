#include "Camera.h"

#include "renderlib/RenderSettings.h"

QCamera::QCamera(QObject* pParent /*= NULL*/)
  : QObject(pParent)
  , m_Film()
  , m_Aperture()
  , m_Projection()
  , m_Focus()
{
}

QCamera::~QCamera() {}

QCamera::QCamera(const QCamera& Other)
{
  *this = Other;
};

QCamera&
QCamera::operator=(const QCamera& Other)
{
  blockSignals(true);

  m_Film = Other.m_Film;
  m_Aperture = Other.m_Aperture;
  m_Projection = Other.m_Projection;
  m_Focus = Other.m_Focus;

  blockSignals(false);

  emit Changed();

  return *this;
}

QFilm&
QCamera::GetFilm()
{
  return m_Film;
}

void
QCamera::SetFilm(const QFilm& Film)
{
  m_Film = Film;
}

QAperture&
QCamera::GetAperture()
{
  return m_Aperture;
}

void
QCamera::SetAperture(const QAperture& Aperture)
{
  m_Aperture = Aperture;
}

QProjection&
QCamera::GetProjection()
{
  return m_Projection;
}

void
QCamera::SetProjection(const QProjection& Projection)
{
  m_Projection = Projection;
}

QFocus&
QCamera::GetFocus()
{
  return m_Focus;
}

void
QCamera::SetFocus(const QFocus& Focus)
{
  m_Focus = Focus;
}

QCamera
QCamera::Default()
{
  QCamera DefaultCamera;

  // DefaultCamera.SetName("Default");

  return DefaultCamera;
}
