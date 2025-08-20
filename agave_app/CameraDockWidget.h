#pragma once

#include <QDockWidget>

#include "CameraWidget.h"

class QCameraDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QCameraDockWidget(QWidget* pParent = NULL, RenderSettings* rs = NULL, CameraDataObject* cdo = NULL);

private:
  QCameraWidget m_CameraWidget;
};
