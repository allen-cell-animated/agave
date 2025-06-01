#pragma once

#include <QDockWidget>

#include "AppearanceWidget.h"

class QAppearanceDockWidget2 : public QDockWidget
{
  Q_OBJECT

public:
  QAppearanceDockWidget2(QWidget* pParent = NULL, RenderSettings* rs = NULL, AppearanceDataObject* cdo = NULL);

private:
  QAppearanceWidget2 m_AppearanceWidget;
};
