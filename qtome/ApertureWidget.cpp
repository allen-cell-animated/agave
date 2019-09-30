#include "ApertureWidget.h"
#include "Camera.h"

#include <QLabel>

QApertureWidget::QApertureWidget(QWidget* pParent, QCamera* cam)
  : QGroupBox(pParent)
  , m_Layout()
  , m_SizeSlider()
  , m_camera(cam)
{
  setTitle("Aperture");
  setStatusTip("Aperture properties");
  setToolTip("Aperture properties");

  Controls::initFormLayout(m_Layout);
  setLayout(&m_Layout);

  // Aperture size
  m_SizeSlider.setRange(0.0, 0.1);
  m_SizeSlider.setSuffix(" mm");
  m_SizeSlider.setDecimals(3);
  m_SizeSlider.setValue(0.0);
  m_SizeSlider.setSingleStep(0.01);
  m_Layout.addRow("Size", &m_SizeSlider);

  connect(&m_SizeSlider, SIGNAL(valueChanged(double)), this, SLOT(SetAperture(double)));
  connect(&cam->GetAperture(), SIGNAL(Changed(const QAperture&)), this, SLOT(OnApertureChanged(const QAperture&)));

  // gStatus.SetStatisticChanged("Camera", "Aperture", "", "", "");
}

void
QApertureWidget::SetAperture(const double& Aperture)
{
  m_camera->GetAperture().SetSize(Aperture);
}

void
QApertureWidget::OnApertureChanged(const QAperture& Aperture)
{
  m_SizeSlider.setValue(Aperture.GetSize(), true);

  // gStatus.SetStatisticChanged("Aperture", "Size", QString::number(Aperture.GetSize(), 'f', 3), "mm");
}
