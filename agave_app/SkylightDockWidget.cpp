#include "SkylightDockWidget.h"

QSkyLightDockWidget::QSkyLightDockWidget(QWidget* pParent, RenderSettings* rs, SkyLightObject* skylightObject)
  : QDockWidget(pParent)
  , m_SkylightWidget(nullptr, rs, skylightObject)
{
  setWindowTitle("Skylight");

  setWidget(&m_SkylightWidget);

  QSizePolicy SizePolicy;

  SizePolicy.setVerticalPolicy(QSizePolicy::Maximum);

  setSizePolicy(SizePolicy);
}
