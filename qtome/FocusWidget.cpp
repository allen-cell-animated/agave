#include "FocusWidget.h"
#include "Camera.h"

#include <QLabel>

QFocusWidget::QFocusWidget(QWidget* pParent, QCamera* cam) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_FocusTypeComboBox(),
	m_FocalDistanceSlider(),
	m_FocalDistanceSpinner(),
	m_qcamera(cam)
{
	setTitle("Focus");
	setStatusTip("Focus properties");
	setToolTip("Focus properties");

	m_GridLayout.setColumnMinimumWidth(0, 75);
	setLayout(&m_GridLayout);

	// Focal distance
	m_GridLayout.addWidget(new QLabel("Focal distance"), 0, 0);

	m_FocalDistanceSlider.setOrientation(Qt::Horizontal);
    m_FocalDistanceSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_FocalDistanceSlider.setRange(0.0, 15.0);
	m_GridLayout.addWidget(&m_FocalDistanceSlider, 0, 1);
	
    m_FocalDistanceSpinner.setRange(0.0, 15.0);
	m_FocalDistanceSpinner.setSuffix(" m");
	m_GridLayout.addWidget(&m_FocalDistanceSpinner, 0, 2);
	
	connect(&m_FocalDistanceSlider, SIGNAL(valueChanged(double)), &m_FocalDistanceSpinner, SLOT(setValue(double)));
	connect(&m_FocalDistanceSlider, SIGNAL(valueChanged(double)), this, SLOT(SetFocalDistance(double)));
	connect(&m_FocalDistanceSpinner, SIGNAL(valueChanged(double)), &m_FocalDistanceSlider, SLOT(setValue(double)));
	connect(&cam->GetFocus(), SIGNAL(Changed(const QFocus&)), this, SLOT(OnFocusChanged(const QFocus&)));

	//gStatus.SetStatisticChanged("Camera", "Focus", "", "", "");
}

void QFocusWidget::SetFocusType(int FocusType)
{
	m_qcamera->GetFocus().SetType(FocusType);
}

void QFocusWidget::SetFocalDistance(const double& FocalDistance)
{
	m_qcamera->GetFocus().SetFocalDistance(FocalDistance);
}

void QFocusWidget::OnFocusChanged(const QFocus& Focus)
{
	m_FocalDistanceSlider.setValue(Focus.GetFocalDistance(), true);
	m_FocalDistanceSpinner.setValue(Focus.GetFocalDistance(), true);

	//gStatus.SetStatisticChanged("Focus", "Focal Distance", QString::number(Focus.GetFocalDistance(), 'f', 2), "mm");
}