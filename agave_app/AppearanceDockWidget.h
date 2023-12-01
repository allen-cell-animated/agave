#pragma once

#include <QDockWidget>

#include "AppearanceSettingsWidget.h"

class QAppearanceDockWidget;
class RenderSettings;

class QAppearanceWidget : public QWidget
{
  Q_OBJECT

public:
  QAppearanceWidget(QWidget* pParent = NULL, QRenderSettings* qrs = nullptr, RenderSettings* rs = nullptr);

  void onNewImage(Scene* s, bool keepCurrentSettings) { m_AppearanceSettingsWidget.onNewImage(s, keepCurrentSettings); }

protected:
  QGridLayout m_MainLayout;
  QAppearanceSettingsWidget m_AppearanceSettingsWidget;
};

class QAppearanceDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QAppearanceDockWidget(QWidget* pParent = NULL, QRenderSettings* qrs = nullptr, RenderSettings* rs = nullptr);

  void onNewImage(Scene* s, bool keepCurrentSettings) { m_VolumeAppearanceWidget.onNewImage(s, keepCurrentSettings); }

protected:
  QAppearanceWidget m_VolumeAppearanceWidget;
};
