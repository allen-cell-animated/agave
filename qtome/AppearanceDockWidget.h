#pragma once

#include <QtWidgets/QDockWidget>

#include "AppearanceSettingsWidget.h"

class QAppearanceDockWidget;
class RenderSettings;

class QAppearanceWidget : public QWidget
{
  Q_OBJECT

public:
  QAppearanceWidget(QWidget* pParent = NULL, QRenderSettings* qrs = nullptr, RenderSettings* rs = nullptr);

  void onNewImage(Scene* s, std::string filepath) { m_AppearanceSettingsWidget.onNewImage(s, filepath); }

protected:
  QGridLayout m_MainLayout;
  QAppearanceSettingsWidget m_AppearanceSettingsWidget;
};

class QAppearanceDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QAppearanceDockWidget(QWidget* pParent = NULL, QRenderSettings* qrs = nullptr, RenderSettings* rs = nullptr);

  void onNewImage(Scene* s, std::string filepath) { m_VolumeAppearanceWidget.onNewImage(s, filepath); }

protected:
  QAppearanceWidget m_VolumeAppearanceWidget;
};
