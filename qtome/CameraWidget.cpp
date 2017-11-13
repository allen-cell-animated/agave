#include "Stable.h"

#include "CameraWidget.h"
#include "RenderThread.h"
#include "Scene.h"

QCameraWidget::QCameraWidget(QWidget* pParent, QCamera* cam, CScene* scene) :
	QWidget(pParent),
	m_MainLayout(),
	m_FilmWidget(nullptr, cam, scene),
	m_ApertureWidget(nullptr, cam),
	m_ProjectionWidget(nullptr, cam),
	m_FocusWidget(nullptr, cam)
{
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	//m_MainLayout.addWidget(&m_PresetsWidget);
	m_MainLayout.addWidget(&m_FilmWidget);
	m_MainLayout.addWidget(&m_ApertureWidget);
	m_MainLayout.addWidget(&m_ProjectionWidget);
//	m_MainLayout.addWidget(&m_FocusWidget);

	QObject::connect(&cam->GetFilm(), SIGNAL(Changed(const QFilm&)), cam, SLOT(OnFilmChanged()));
	QObject::connect(&cam->GetAperture(), SIGNAL(Changed(const QAperture&)), cam, SLOT(OnApertureChanged()));
	QObject::connect(&cam->GetProjection(), SIGNAL(Changed(const QProjection&)), cam, SLOT(OnProjectionChanged()));
	QObject::connect(&cam->GetFocus(), SIGNAL(Changed(const QFocus&)), cam, SLOT(OnFocusChanged()));
	//QObject::connect(&m_PresetsWidget, SIGNAL(LoadPreset(const QString&)), this, SLOT(OnLoadPreset(const QString&)));
	//QObject::connect(&m_PresetsWidget, SIGNAL(SavePreset(const QString&)), this, SLOT(OnSavePreset(const QString&)));
	//QObject::connect(&gStatus, SIGNAL(LoadPreset(const QString&)), &m_PresetsWidget, SLOT(OnLoadPreset(const QString&)));
}

QSize QCameraWidget::sizeHint() const
{
	return QSize(20, 20);
}