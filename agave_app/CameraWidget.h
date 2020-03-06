#pragma once

#include "ApertureWidget.h"
#include "Camera.h"
#include "FilmWidget.h"
#include "FocusWidget.h"
#include "ProjectionWidget.h"

class RenderSettings;

class QCameraWidget : public QWidget
{
  Q_OBJECT

public:
  QCameraWidget(QWidget* pParent = NULL, QCamera* cam = nullptr, RenderSettings* rs = nullptr);

  virtual QSize sizeHint() const;

private:
  QGridLayout m_MainLayout;
  QFilmWidget m_FilmWidget;
  QApertureWidget m_ApertureWidget;
  QProjectionWidget m_ProjectionWidget;
  QFocusWidget m_FocusWidget;
};
