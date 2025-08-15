#include "ArealightWidget.h"
#include "RenderSettings.h"

#include "qtControls/controlFactory.h"

#include "renderlib/uiInfo.hpp"
#include "renderlib/AreaLightObject.hpp"
#include "renderlib/ViewerWindow.h"

#include <QLabel>

QAreaLightWidget::QAreaLightWidget(QWidget* pParent, RenderSettings* rs, AreaLightObject* arealightObject)
  : QWidget(pParent)
  , m_MainLayout()
  , m_renderSettings(rs)
  , m_arealightObject(arealightObject)
{
  Controls::initFormLayout(m_MainLayout);
  setLayout(&m_MainLayout);

  if (m_arealightObject) {
    createFlatList(&m_MainLayout, m_arealightObject);
  }
}

QSize
QAreaLightWidget::sizeHint() const
{
  return QSize(20, 20);
}
