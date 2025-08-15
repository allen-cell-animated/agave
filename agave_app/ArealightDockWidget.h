#pragma once

#include <QDockWidget>

#include "ArealightWidget.h"

class QArealightDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QArealightDockWidget(QWidget* pParent = NULL,
                       RenderSettings* rs = NULL,
                       ViewerWindow* vw = NULL,
                       ArealightObject* arealightObject = NULL);

private:
  QArealightWidget m_ArealightWidget;
};
