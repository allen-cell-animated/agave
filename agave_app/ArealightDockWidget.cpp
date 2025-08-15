#include "ArealightDockWidget.h"

QArealightDockWidget::QArealightDockWidget(QWidget* pParent,
                                           RenderSettings* rs,
                                           ViewerWindow* vw,
                                           ArealightObject* arealightObject)
  : QDockWidget(pParent)
  , m_ArealightWidget(nullptr, rs, vw, arealightObject)
{
  setWindowTitle("Area Light");

  setWidget(&m_ArealightWidget);

  QSizePolicy SizePolicy;

  SizePolicy.setVerticalPolicy(QSizePolicy::Maximum);

  setSizePolicy(SizePolicy);
}
