#pragma once

#include "Camera.h"
#include "qtControls/Controls.h"

// #include "renderlib/core/prty/prtyProperty.h"
#include "renderlib/CameraDataObject.hpp"
#include "renderlib/Logging.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFormLayout>
#include <QWidget>

class RenderSettings;

class QCameraWidget : public QWidget
{
  Q_OBJECT

public:
  QCameraWidget(QWidget* pParent = NULL,
                QCamera* cam = nullptr,
                RenderSettings* rs = nullptr,
                CameraDataObject* cdo = nullptr);

  virtual QSize sizeHint() const;

private:
  QFormLayout m_MainLayout;

  RenderSettings* m_renderSettings;

private:
  CameraDataObject* m_cameraDataObject;
};
