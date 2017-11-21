#include "Stable.h"

#include "CameraDockWidget.h"

QCameraDockWidget::QCameraDockWidget(QWidget* pParent, QCamera* cam, CScene* scene) :
	QDockWidget(pParent),
	m_CameraWidget(nullptr, cam, scene)
{
	setWindowTitle("Camera");
	setToolTip("<img src=':/Images/camera.png'><div>Camera Properties</div>");
	setWindowIcon(GetIcon("camera"));

	setWidget(&m_CameraWidget);

	QSizePolicy SizePolicy;

	SizePolicy.setVerticalPolicy(QSizePolicy::Maximum);

	setSizePolicy(SizePolicy);
}