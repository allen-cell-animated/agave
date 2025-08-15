#include "SkylightDockWidget.h"

QSkylightDockWidget::QSkylightDockWidget(QWidget* pParent,
                                         RenderSettings* rs,
                                         ViewerWindow* vw,
                                         SkylightObject* skylightObject)
  : QDockWidget(pParent)
  , m_SkylightWidget(nullptr, rs, vw, skylightObject)
{
  setWindowTitle("Skylight");

  setWidget(&m_SkylightWidget);

  QSizePolicy SizePolicy;

  SizePolicy.setVerticalPolicy(QSizePolicy::Maximum);

  setSizePolicy(SizePolicy);
}
