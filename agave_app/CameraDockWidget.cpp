#include "CameraDockWidget.h"

QCameraDockWidget::QCameraDockWidget(QWidget* pParent, QCamera* cam, RenderSettings* rs, CameraDataObject* cdo)
  : QDockWidget(pParent)
  , m_CameraWidget(nullptr, cam, rs, cdo)
{
  setWindowTitle("Camera");

  setWidget(&m_CameraWidget);

  QSizePolicy SizePolicy;

  SizePolicy.setVerticalPolicy(QSizePolicy::Maximum);

  setSizePolicy(SizePolicy);
}
