#include "FocusWidget.h"
#include "Camera.h"

#include <QLabel>

QFocusWidget::QFocusWidget(QWidget* pParent, QCamera* cam)
  : QGroupBox(pParent)
  , m_Layout()
  , m_FocusTypeComboBox()
  , m_FocalDistanceSlider()
  , m_qcamera(cam)
{
  setTitle("Focus");
  setStatusTip(tr("Focus properties"));
  setToolTip(tr("Focus properties"));

  Controls::initFormLayout(m_Layout);
  setLayout(&m_Layout);

  // Focal distance
  m_FocalDistanceSlider.setStatusTip(tr("Set focal distance"));
  m_FocalDistanceSlider.setToolTip(tr("Set focal distance"));
  m_FocalDistanceSlider.setRange(0.0, 15.0);
  m_FocalDistanceSlider.setValue(0.0);
  m_FocalDistanceSlider.setSuffix(" m");

  m_Layout.addRow("Focal distance", &m_FocalDistanceSlider);

  QObject::connect(&m_FocalDistanceSlider, &QNumericSlider::valueChanged, this, &QFocusWidget::SetFocalDistance);

  connect(&cam->GetFocus(), SIGNAL(Changed(const QFocus&)), this, SLOT(OnFocusChanged(const QFocus&)));

  // gStatus.SetStatisticChanged("Camera", "Focus", "", "", "");
}

void
QFocusWidget::SetFocusType(int FocusType)
{
  m_qcamera->GetFocus().SetType(FocusType);
}

void
QFocusWidget::SetFocalDistance(const double& FocalDistance)
{
  m_qcamera->GetFocus().SetFocalDistance(FocalDistance);
}

void
QFocusWidget::OnFocusChanged(const QFocus& Focus)
{
  m_FocalDistanceSlider.setValue(Focus.GetFocalDistance(), true);
  // gStatus.SetStatisticChanged("Focus", "Focal Distance", QString::number(Focus.GetFocalDistance(), 'f', 2), "mm");
}
