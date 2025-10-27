#pragma once

#include "qtControls/Controls.h"

#include "renderlib/AreaLightObject.hpp"
#include "renderlib/Logging.h"

#include <QFormLayout>
#include <QWidget>

class RenderSettings;
class ViewerWindow;

class QAreaLightWidget : public QWidget
{
  Q_OBJECT

public:
  QAreaLightWidget(QWidget* pParent = NULL, RenderSettings* rs = nullptr, AreaLightObject* arealightObject = nullptr);

  virtual QSize sizeHint() const;

private:
  QFormLayout m_MainLayout;
  RenderSettings* m_renderSettings;
  AreaLightObject* m_arealightObject;
};
