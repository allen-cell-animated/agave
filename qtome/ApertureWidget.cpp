#include "ApertureWidget.h"
#include "Camera.h"

#include <QLabel>

QApertureWidget::QApertureWidget(QWidget* pParent, QCamera* cam) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_SizeSlider(),
	m_camera(cam)
{
	setTitle("Aperture");
	setStatusTip("Aperture properties");
	setToolTip("Aperture properties");

	m_GridLayout.setColumnMinimumWidth(0, 75);
	setLayout(&m_GridLayout);

	// Aperture size
	m_GridLayout.addWidget(new QLabel("Size"), 3, 0);

	m_SizeSlider.setRange(0.0, 0.1);
	m_SizeSlider.setSuffix(" mm");
	m_SizeSlider.setDecimals(3);
	m_SizeSlider.setValue(0.0);
	m_GridLayout.addWidget(&m_SizeSlider, 3, 1, 1, 2);
	
	connect(&m_SizeSlider, SIGNAL(valueChanged(double)), this, SLOT(SetAperture(double)));
	connect(&cam->GetAperture(), SIGNAL(Changed(const QAperture&)), this, SLOT(OnApertureChanged(const QAperture&)));

	//gStatus.SetStatisticChanged("Camera", "Aperture", "", "", "");
}

void QApertureWidget::SetAperture(const double& Aperture)
{
	m_camera->GetAperture().SetSize(Aperture);
}

void QApertureWidget::OnApertureChanged(const QAperture& Aperture)
{
 	m_SizeSlider.setValue(Aperture.GetSize(), true);

	//gStatus.SetStatisticChanged("Aperture", "Size", QString::number(Aperture.GetSize(), 'f', 3), "mm");
}