#include "CameraDockWidget.h"

QCameraDockWidget::QCameraDockWidget(QWidget* pParent, RenderSettings* rs, CameraDataObject* cdo)
  : QDockWidget(pParent)
  , m_CameraWidget(nullptr, rs, cdo)
{
  setWindowTitle("Camera");

  setWidget(&m_CameraWidget);

  QSizePolicy SizePolicy;

  SizePolicy.setVerticalPolicy(QSizePolicy::Maximum);

  setSizePolicy(SizePolicy);
}
