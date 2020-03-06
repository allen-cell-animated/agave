#pragma once

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

  void onNewImage(Scene* s, std::string filepath);

  void OnTimeChanged(int newTime);

protected:
  QGridLayout m_MainLayout;
  QIntSlider* m_TimeSlider;

  QRenderSettings* m_qrendersettings;
  Scene* m_scene;
  std::string m_filepath;
};

class QTimelineDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QTimelineDockWidget(QWidget* pParent = NULL, QRenderSettings* qrs = nullptr);

  void onNewImage(Scene* s, std::string filepath) { m_TimelineWidget.onNewImage(s, filepath); }

protected:
  QTimelineWidget m_TimelineWidget;
};
