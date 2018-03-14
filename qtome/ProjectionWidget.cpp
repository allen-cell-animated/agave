#include "Stable.h"

#include "ProjectionWidget.h"
#include "RenderThread.h"
#include "Camera.h"

#include "renderlib/RenderSettings.h"

QProjectionWidget::QProjectionWidget(QWidget* pParent, QCamera* cam, RenderSettings* rs) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_FieldOfViewSlider(),
	_camera(cam)
{
	setTitle("Projection");
	setStatusTip("Projection properties");
	setToolTip("Projection properties");

	m_GridLayout.setColumnMinimumWidth(0, 75);
	setLayout(&m_GridLayout);

	// Field of view
	m_GridLayout.addWidget(new QLabel("Field of view"), 4, 0);

	m_FieldOfViewSlider.setRange(10.0, 150.0);
	m_FieldOfViewSlider.setValue(cam->GetProjection().GetFieldOfView());
	m_FieldOfViewSlider.setSuffix(" deg.");
	m_GridLayout.addWidget(&m_FieldOfViewSlider, 4, 1, 1, 2);
	
	connect(&m_FieldOfViewSlider, SIGNAL(valueChanged(double)), this, SLOT(SetFieldOfView(double)));
	connect(&cam->GetProjection(), SIGNAL(Changed(const QProjection&)), this, SLOT(OnProjectionChanged(const QProjection&)));

	//gStatus.SetStatisticChanged("Camera", "Projection", "", "", "");
}

void QProjectionWidget::SetFieldOfView(const double& FieldOfView)
{
	_camera->GetProjection().SetFieldOfView(FieldOfView);
}

void QProjectionWidget::OnProjectionChanged(const QProjection& Projection)
{
	m_FieldOfViewSlider.setValue(Projection.GetFieldOfView(), true);

	//gStatus.SetStatisticChanged("Projection", "Field Of View", QString::number(Projection.GetFieldOfView(), 'f', 2), "Deg.");
}