#include "Stable.h"

#include "CameraWidget.h"
#include "RenderThread.h"
#include "RenderSettings.h"

QCameraWidget::QCameraWidget(QWidget* pParent, QCamera* cam, RenderSettings* rs) :
	QWidget(pParent),
	m_MainLayout(),
	m_FilmWidget(nullptr, cam, rs),
	m_ApertureWidget(nullptr, cam),
	m_ProjectionWidget(nullptr, cam, rs),
	m_FocusWidget(nullptr, cam)
{
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	//m_MainLayout.addWidget(&m_PresetsWidget);
	m_MainLayout.addWidget(&m_FilmWidget);
	m_MainLayout.addWidget(&m_ApertureWidget);
	m_MainLayout.addWidget(&m_ProjectionWidget);
	m_MainLayout.addWidget(&m_FocusWidget);

	QObject::connect(&cam->GetFilm(), SIGNAL(Changed(const QFilm&)), cam, SLOT(OnFilmChanged()));
	QObject::connect(&cam->GetAperture(), SIGNAL(Changed(const QAperture&)), cam, SLOT(OnApertureChanged()));
	QObject::connect(&cam->GetProjection(), SIGNAL(Changed(const QProjection&)), cam, SLOT(OnProjectionChanged()));
	QObject::connect(&cam->GetFocus(), SIGNAL(Changed(const QFocus&)), cam, SLOT(OnFocusChanged()));
}

QSize QCameraWidget::sizeHint() const
{
	return QSize(20, 20);
}