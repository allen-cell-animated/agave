#pragma once

#include "Controls.h"
#include "Focus.h"

#include <QGroupBox>

class QCamera;

class QFocusWidget : public QGroupBox
{
  Q_OBJECT

public:
  QFocusWidget(QWidget* pParent = NULL, QCamera* cam = nullptr);

private slots:
  void SetFocusType(int FocusType);
  void SetFocalDistance(const double& FocalDistance);
  void OnFocusChanged(const QFocus& Focus);

private:
  QGridLayout m_GridLayout;
  QComboBox m_FocusTypeComboBox;
  QDoubleSlider m_FocalDistanceSlider;
  QDoubleSpinner m_FocalDistanceSpinner;
  QCamera* m_qcamera;
};
