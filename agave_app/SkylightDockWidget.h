#pragma once

#include <QDockWidget>

#include "SkylightWidget.h"

class QSkyLightDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QSkyLightDockWidget(QWidget* pParent = NULL, RenderSettings* rs = NULL, SkyLightObject* skylightObject = NULL);

private:
  QSkyLightWidget m_SkylightWidget;
};
