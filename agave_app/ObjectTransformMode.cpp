#include "ObjectTransformMode.h"
#include "QRenderSettings.h"

#include "renderlib/AppScene.h"

ObjectTransformMode::ObjectTransformMode(std::function<Scene*()> getScene, QRenderSettings* qrs, QObject* parent)
  : QObject(parent)
  , m_getScene(std::move(getScene))
  , m_qrendersettings(qrs)
{
}

void
ObjectTransformMode::registerButton(QPushButton* button, QAction* action, std::function<SceneObject*()> getObject)
{
  button->setCheckable(true);

  Entry entry{ button, action, std::move(getObject) };
  m_entries.push_back(entry);

  QObject::connect(button, &QPushButton::clicked, [this, idx = m_entries.size() - 1]() { activate(m_entries[idx]); });
}

void
ObjectTransformMode::activate(Entry& entry)
{
  Scene* scene = m_getScene();
  if (!scene) {
    return;
  }

  SceneObject* object = entry.getObject();
  if (!object) {
    return;
  }

  bool wasActionOn = entry.action->isChecked();
  bool isObjectSelected = scene->m_selection == object;

  if (!wasActionOn) {
    // Turning the action on: select the object and trigger the action.
    emit m_qrendersettings->Selected(object);
    entry.action->trigger();
  } else {
    // Action is already on.
    if (!isObjectSelected) {
      // Different object selected — just switch selection to this object.
      emit m_qrendersettings->Selected(object);
    } else {
      // Same object already selected — deselect and turn action off.
      emit m_qrendersettings->Selected(nullptr);
      entry.action->trigger();
    }
  }

  updateButtonStates();
}

void
ObjectTransformMode::updateButtonStates()
{
  Scene* scene = m_getScene();

  for (auto& entry : m_entries) {
    bool checked = false;
    if (scene && entry.action->isChecked()) {
      SceneObject* obj = entry.getObject();
      if (obj && scene->m_selection == obj) {
        checked = true;
      }
    }
    entry.button->setChecked(checked);
  }
}
