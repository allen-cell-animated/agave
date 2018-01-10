#pragma once

#include <QtWidgets/QDockWidget>

#include "CameraWidget.h"

class QCameraDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QCameraDockWidget(QWidget* pParent = NULL, QCamera* cam = NULL, CScene* scene = NULL);

private:
	QCameraWidget	m_CameraWidget;
};