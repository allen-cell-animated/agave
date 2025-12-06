#pragma once

#include "qtControls/Controls.h"

#include "renderlib/SkyLightObject.hpp"
#include "renderlib/Logging.h"

#include <QFormLayout>
#include <QWidget>

class RenderSettings;
class ViewerWindow;

class QSkyLightWidget : public QWidget
{
  Q_OBJECT

public:
  QSkyLightWidget(QWidget* pParent = NULL, RenderSettings* rs = nullptr, SkyLightObject* skylightObject = nullptr);

  virtual QSize sizeHint() const;

private:
  QFormLayout m_MainLayout;
  RenderSettings* m_renderSettings;
  SkyLightObject* m_skylightObject;
};
