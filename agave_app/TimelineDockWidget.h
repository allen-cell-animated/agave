#pragma once

#include "renderlib/io/FileReader.h"

#include <QDockWidget>
#include <QGridLayout>

class IFileReader;
class QIntSlider;
class QTimelineDockWidget;
class QRenderSettings;
class Scene;

class QTimelineWidget : public QWidget
{
  Q_OBJECT

public:
  QTimelineWidget(QWidget* pParent = NULL, QRenderSettings* qrs = nullptr);

  void onNewImage(Scene* s, const LoadSpec& loadSpec, std::shared_ptr<IFileReader> reader);
  void setTime(int t);

  void OnTimeChanged(int newTime);

signals:
  void timeChanged(int newTime);

protected:
  QGridLayout m_MainLayout;
  QIntSlider* m_TimeSlider;

  QRenderSettings* m_qrendersettings;
  Scene* m_scene;
  LoadSpec m_loadSpec;
  std::shared_ptr<IFileReader> m_reader;
};

class QTimelineDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QTimelineDockWidget(QWidget* pParent = NULL, QRenderSettings* qrs = nullptr);

  void onNewImage(Scene* s, const LoadSpec& loadSpec, std::shared_ptr<IFileReader> reader)
  {
    m_TimelineWidget.onNewImage(s, loadSpec, reader);
  }
  void setTime(int t) { m_TimelineWidget.setTime(t); }

  // other Gui needs to connect to timeline signals
  QTimelineWidget& timelineWidget() { return m_TimelineWidget; }

protected:
  QTimelineWidget m_TimelineWidget;
};
