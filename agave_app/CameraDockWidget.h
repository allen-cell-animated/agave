#pragma once

#include <QDockWidget>

#include "CameraWidget.h"

class QCameraDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QCameraDockWidget(QWidget* pParent = nullptr, QCamera* cam = nullptr, RenderSettings* rs = nullptr);

private:
  QCameraWidget m_CameraWidget;
};
