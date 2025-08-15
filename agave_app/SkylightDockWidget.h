#pragma once

#include <QDockWidget>

#include "SkylightWidget.h"

class QSkylightDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QSkylightDockWidget(QWidget* pParent = NULL,
                      RenderSettings* rs = NULL,
                      ViewerWindow* vw = NULL,
                      SkylightObject* skylightObject = NULL);

private:
  QSkylightWidget m_SkylightWidget;
};
