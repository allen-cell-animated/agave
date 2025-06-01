#include "AppearanceDockWidget2.h"

QAppearanceDockWidget2::QAppearanceDockWidget2(QWidget* pParent, RenderSettings* rs, AppearanceDataObject* ado)
  : QDockWidget(pParent)
  , m_AppearanceWidget(nullptr, rs, ado)
{
  setWindowTitle("Appearance");

  setWidget(&m_AppearanceWidget);

  QSizePolicy SizePolicy;

  SizePolicy.setVerticalPolicy(QSizePolicy::Maximum);

  setSizePolicy(SizePolicy);
}
