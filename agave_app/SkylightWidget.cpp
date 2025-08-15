#include "SkylightWidget.h"
#include "RenderSettings.h"

#include "qtControls/controlFactory.h"

#include "renderlib/uiInfo.hpp"
#include "renderlib/SkylightObject.hpp"
#include "renderlib/ViewerWindow.h"

#include <QLabel>

QSkylightWidget::QSkylightWidget(QWidget* pParent, RenderSettings* rs, ViewerWindow* vw, SkylightObject* skylightObject)
  : QWidget(pParent)
  , m_MainLayout()
  , m_renderSettings(rs)
  , m_skylightObject(skylightObject)
{
  Controls::initFormLayout(m_MainLayout);
  setLayout(&m_MainLayout);

  if (m_skylightObject) {
    createFlatList(&m_MainLayout, m_skylightObject);
  }
}

QSize
QSkylightWidget::sizeHint() const
{
  return QSize(20, 20);
}
