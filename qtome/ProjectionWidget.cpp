#include "ProjectionWidget.h"
#include "Camera.h"

#include "renderlib/RenderSettings.h"

#include <QLabel>

QProjectionWidget::QProjectionWidget(QWidget* pParent, QCamera* cam, RenderSettings* rs)
  : QGroupBox(pParent)
  , m_Layout()
  , m_FieldOfViewSlider()
  , m_qcamera(cam)
{
  setTitle("Projection");
  setStatusTip("Projection properties");
  setToolTip("Projection properties");

  Controls::initFormLayout(m_Layout);
  setLayout(&m_Layout);

  // Field of view
  m_FieldOfViewSlider.setRange(10.0, 150.0);
  m_FieldOfViewSlider.setValue(cam->GetProjection().GetFieldOfView());
  m_FieldOfViewSlider.setSuffix(" deg.");
  m_Layout.addRow("Field of view", &m_FieldOfViewSlider);

  connect(&m_FieldOfViewSlider, SIGNAL(valueChanged(double)), this, SLOT(SetFieldOfView(double)));
  connect(
    &cam->GetProjection(), SIGNAL(Changed(const QProjection&)), this, SLOT(OnProjectionChanged(const QProjection&)));

  // gStatus.SetStatisticChanged("Camera", "Projection", "", "", "");
}

void
QProjectionWidget::SetFieldOfView(const double& FieldOfView)
{
  m_qcamera->GetProjection().SetFieldOfView(FieldOfView);
}

void
QProjectionWidget::OnProjectionChanged(const QProjection& Projection)
{
  m_FieldOfViewSlider.setValue(Projection.GetFieldOfView(), true);

  // gStatus.SetStatisticChanged("Projection", "Field Of View", QString::number(Projection.GetFieldOfView(), 'f', 2),
  // "Deg.");
}
