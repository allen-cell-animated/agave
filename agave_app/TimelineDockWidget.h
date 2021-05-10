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

  void onNewImage(Scene* s, std::string filepath, int sceneIndex);

  void OnTimeChanged(int newTime);

protected:
  QGridLayout m_MainLayout;
  QIntSlider* m_TimeSlider;

  QRenderSettings* m_qrendersettings;
  Scene* m_scene;
  std::string m_filepath;
  int m_currentScene;
};

class QTimelineDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QTimelineDockWidget(QWidget* pParent = NULL, QRenderSettings* qrs = nullptr);

  void onNewImage(Scene* s, std::string filepath, int sceneIndex) { m_TimelineWidget.onNewImage(s, filepath, sceneIndex); }

protected:
  QTimelineWidget m_TimelineWidget;
};
