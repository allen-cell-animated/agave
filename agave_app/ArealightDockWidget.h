#pragma once

#include <QDockWidget>

#include "ArealightWidget.h"

class QAreaLightDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QAreaLightDockWidget(QWidget* pParent = NULL, RenderSettings* rs = NULL, AreaLightObject* arealightObject = NULL);

private:
  QAreaLightWidget m_ArealightWidget;
};
