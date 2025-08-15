#include "ArealightWidget.h"
#include "RenderSettings.h"

#include "qtControls/controlFactory.h"

#include "renderlib/uiInfo.hpp"
#include "renderlib/ArealightObject.hpp"
#include "renderlib/ViewerWindow.h"

#include <QLabel>

QArealightWidget::QArealightWidget(QWidget* pParent,
                                   RenderSettings* rs,
                                   ViewerWindow* vw,
                                   ArealightObject* arealightObject)
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
QArealightWidget::sizeHint() const
{
  return QSize(20, 20);
}
