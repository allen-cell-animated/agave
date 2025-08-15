#pragma once

#include "qtControls/Controls.h"

#include "renderlib/ArealightObject.hpp"
#include "renderlib/Logging.h"

#include <QFormLayout>
#include <QWidget>

class RenderSettings;
class ViewerWindow;

class QArealightWidget : public QWidget
{
  Q_OBJECT

public:
  QArealightWidget(QWidget* pParent = NULL,
                   RenderSettings* rs = nullptr,
                   ViewerWindow* vw = nullptr,
                   ArealightObject* arealightObject = nullptr);

  virtual QSize sizeHint() const;

private:
  QFormLayout m_MainLayout;
  RenderSettings* m_renderSettings;
  ArealightObject* m_arealightObject;
};
