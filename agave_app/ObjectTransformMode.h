#pragma once

#include <QObject>
#include <QPushButton>

#include <functional>
#include <vector>

class QAction;
class QRenderSettings;
class Scene;
class SceneObject;

class ObjectTransformMode : public QObject
{
  Q_OBJECT

public:
  ObjectTransformMode(std::function<Scene*()> getScene, QRenderSettings* qrs, QObject* parent = nullptr);

  void registerButton(QPushButton* button, QAction* action, std::function<SceneObject*()> getObject);

  // Turn off any active transform mode (e.g. when the scene is rebuilt and any
  // previously selected SceneObject is about to become invalid). Unchecks any
  // currently-active action and clears button states.
  void clearMode();

  void updateButtonStates();

private:
  struct Entry
  {
    QPushButton* button;
    QAction* action;
    std::function<SceneObject*()> getObject;
  };

  void activate(Entry& entry);

  std::function<Scene*()> m_getScene;
  QRenderSettings* m_qrendersettings;
  std::vector<Entry> m_entries;
};
