#pragma once

#include <QDockWidget>

#include "AppearanceWidget.h"

class QAppearanceDockWidget2 : public QDockWidget
{
  Q_OBJECT

public:
  QAppearanceDockWidget2(QWidget* pParent = NULL,
                         RenderSettings* rs = NULL,
                         ViewerWindow* vw = NULL,
                         AppearanceObject* cdo = NULL);

private:
  QAppearanceWidget2 m_AppearanceWidget;
};
