#include "CameraWidget.h"
#include "RenderSettings.h"

#include "qtControls/controlFactory.h"
#include "qtControls/Section.h"

#include "renderlib/uiInfo.hpp"
#include "renderlib/CameraObject.hpp"

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
  // // loop over all properties in cameraobject. for each property, add a callback that updates the rendersetttings
  // // cameradirty flags
  // for (const auto& prop : m_cameraObject->GetList()) {
  //   if (prop) {
  //     prop->GetProperty(0)->AddCallback(new prtyCallbackLambda([this](prtyProperty* i_Property, bool i_bDirty) {
  //       if (i_bDirty) {
  //         m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
  //       }
  //     }));
  //   }
  // }
}

QSize
QCameraWidget::sizeHint() const
{
  return QSize(20, 20);
}
