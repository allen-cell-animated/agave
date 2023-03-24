#pragma once

#include "renderlib/FileReader.h"

#include <QGridLayout>
#include <QtWidgets/QDockWidget>

class QIntSlider;
class QTimelineDockWidget;
class QRenderSettings;
class Scene;

class QTimelineWidget : public QWidget
{
  Q_OBJECT

public:
  QTimelineWidget(QWidget* pParent = NULL, QRenderSettings* qrs = nullptr);

  void onNewImage(Scene* s, const LoadSpec& loadSpec);
  void setTime(int t);

  void OnTimeChanged(int newTime);

protected:
  QGridLayout m_MainLayout;
  QIntSlider* m_TimeSlider;

  QRenderSettings* m_qrendersettings;
  Scene* m_scene;
  LoadSpec m_loadSpec;
};

class QTimelineDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QTimelineDockWidget(QWidget* pParent = NULL, QRenderSettings* qrs = nullptr);

  void onNewImage(Scene* s, const LoadSpec& loadSpec) { m_TimelineWidget.onNewImage(s, loadSpec); }
  void setTime(int t) { m_TimelineWidget.setTime(t); }

protected:
  QTimelineWidget m_TimelineWidget;
};
