#pragma once

#include "Controls.h"
#include "Projection.h"

class QCamera;
class RenderSettings;

class QProjectionWidget : public QGroupBox
{
    Q_OBJECT

public:
    QProjectionWidget(QWidget* pParent = NULL, QCamera* cam = nullptr, RenderSettings* rs = nullptr);

private slots:
	void SetFieldOfView(const double& FieldOfView);
	void OnProjectionChanged(const QProjection& Film);

private:
	QGridLayout		m_GridLayout;
	QNumericSlider	m_FieldOfViewSlider;

	QCamera* _camera;
};