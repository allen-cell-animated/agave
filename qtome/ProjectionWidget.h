#pragma once

#include "Controls.h"
#include "Projection.h"

class QCamera;
class CScene;

class QProjectionWidget : public QGroupBox
{
    Q_OBJECT

public:
    QProjectionWidget(QWidget* pParent = NULL, QCamera* cam = nullptr, CScene* scene = nullptr);

private slots:
	void SetFieldOfView(const double& FieldOfView);
	void OnProjectionChanged(const QProjection& Film);

private:
	QGridLayout		m_GridLayout;
	QDoubleSlider	m_FieldOfViewSlider;
	QDoubleSpinner	m_FieldOfViewSpinner;

	QCamera* _camera;
};