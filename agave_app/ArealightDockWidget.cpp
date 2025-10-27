#include "ArealightDockWidget.h"

QAreaLightDockWidget::QAreaLightDockWidget(QWidget* pParent, RenderSettings* rs, AreaLightObject* arealightObject)
  : QDockWidget(pParent)
  , m_ArealightWidget(nullptr, rs, arealightObject)
{
  setWindowTitle("Area Light");

  setWidget(&m_ArealightWidget);

  QSizePolicy SizePolicy;

  SizePolicy.setVerticalPolicy(QSizePolicy::Maximum);

  setSizePolicy(SizePolicy);
}
