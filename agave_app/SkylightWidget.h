#pragma once

#include "qtControls/Controls.h"

#include "renderlib/SkylightObject.hpp"
#include "renderlib/Logging.h"

#include <QFormLayout>
#include <QWidget>

class RenderSettings;
class ViewerWindow;

class QSkylightWidget : public QWidget
{
  Q_OBJECT

public:
  QSkylightWidget(QWidget* pParent = NULL,
                  RenderSettings* rs = nullptr,
                  ViewerWindow* vw = nullptr,
                  SkylightObject* skylightObject = nullptr);

  virtual QSize sizeHint() const;

private:
  QFormLayout m_MainLayout;
  RenderSettings* m_renderSettings;
  SkylightObject* m_skylightObject;
};
