#include "Stable.h"

#include "CameraDockWidget.h"

QCameraDockWidget::QCameraDockWidget(QWidget* pParent, QCamera* cam) :
	QDockWidget(pParent),
	m_CameraWidget(nullptr, cam)
{
	setWindowTitle("Camera");
	setToolTip("<img src=':/Images/camera.png'><div>Camera Properties</div>");
	setWindowIcon(GetIcon("camera"));

	setWidget(&m_CameraWidget);

	QSizePolicy SizePolicy;

	SizePolicy.setVerticalPolicy(QSizePolicy::Maximum);

	setSizePolicy(SizePolicy);
}