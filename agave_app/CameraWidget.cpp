#include "CameraWidget.h"
#include "RenderSettings.h"

#include "qtControls/controlFactory.h"
#include "qtControls/Section.h"

#include "renderlib/uiInfo.hpp"

#include <QLabel>
#include <QFormLayout>
#include <map>

QCameraWidget::QCameraWidget(QWidget* pParent, RenderSettings* rs, CameraObject* cameraObject)
  : QWidget(pParent)
  , m_MainLayout()
  , m_renderSettings(rs)
  , m_cameraObject(cameraObject)
{
  Controls::initFormLayout(m_MainLayout);
  setLayout(&m_MainLayout);

  if (m_cameraObject) {
    createFlatList(&m_MainLayout, m_cameraObject);
  }
}

QSize
QCameraWidget::sizeHint() const
{
  return QSize(20, 20);
}
