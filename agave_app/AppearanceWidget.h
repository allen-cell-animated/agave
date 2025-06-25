#pragma once

#include "qtControls/Controls.h"

// #include "renderlib/core/prty/prtyProperty.h"
#include "renderlib/AppearanceUiDescription.hpp"
#include "renderlib/Logging.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFormLayout>
#include <QWidget>

class RenderSettings;
class ViewerWindow;

class QAppearanceWidget2 : public QWidget
{
  Q_OBJECT

public:
  QAppearanceWidget2(QWidget* pParent = NULL,
                     RenderSettings* rs = nullptr,
                     ViewerWindow* vw = nullptr,
                     AppearanceObject* cdo = nullptr);

  virtual QSize sizeHint() const;

private:
  QFormLayout m_MainLayout;

  RenderSettings* m_renderSettings;

private:
  AppearanceObject* m_appearanceDataObject;
};
