#include "ClipPlaneObject.hpp"

#include "ScenePlane.h"
#include "MathUtil.h"
#include "Logging.h"

#include "serialize/docReader.h"
#include "serialize/docWriter.h"

ClipPlaneObject::ClipPlaneObject()
  : prtyObject()
{
  m_scenePlane = std::make_shared<ScenePlane>(glm::vec3(0.0f, 0.0f, 0.0f));

  m_enabledUIInfo = new CheckBoxUiInfo(&m_clipPlaneDataObject.Enabled, "General", "Enabled");
  m_enabledUIInfo->SetToolTip("Enable or disable the clip plane");
  m_enabledUIInfo->SetStatusTip("Enable or disable the clip plane in the scene");
  AddProperty(m_enabledUIInfo);
  m_showHelperUIInfo = new CheckBoxUiInfo(&m_clipPlaneDataObject.ShowHelper, "General", "Show Helper");
  m_showHelperUIInfo->SetToolTip("Show or hide the clip plane helper");
  m_showHelperUIInfo->SetStatusTip("Show or hide the clip plane helper in the scene");
  AddProperty(m_showHelperUIInfo);

  // Add callbacks for property changes
  m_clipPlaneDataObject.Enabled.AddCallback(
    new prtyCallbackWrapper<ClipPlaneObject>(this, &ClipPlaneObject::EnabledChanged));
  m_clipPlaneDataObject.ShowHelper.AddCallback(
    new prtyCallbackWrapper<ClipPlaneObject>(this, &ClipPlaneObject::ShowHelperChanged));
  m_clipPlaneDataObject.Position.AddCallback(
    new prtyCallbackWrapper<ClipPlaneObject>(this, &ClipPlaneObject::PositionChanged));
  m_clipPlaneDataObject.Rotation.AddCallback(
    new prtyCallbackWrapper<ClipPlaneObject>(this, &ClipPlaneObject::RotationChanged));
}

void
ClipPlaneObject::updatePropsFromObject()
{
  if (!m_scenePlane)
    return;

  const Plane& plane = m_scenePlane->m_plane;

  m_clipPlaneDataObject.Enabled.SetValue(m_scenePlane->m_enabled);
  m_clipPlaneDataObject.ShowHelper.SetValue(m_scenePlane->getVisible());
  m_clipPlaneDataObject.Position.SetValue(m_scenePlane->m_transform.m_center);
  m_clipPlaneDataObject.Rotation.SetValue(m_scenePlane->m_transform.m_rotation);
}

void
ClipPlaneObject::updateObjectFromProps()
{
  if (!m_scenePlane)
    return;

  Plane& plane = m_scenePlane->m_plane;

  // update plane
  m_scenePlane->m_enabled = m_clipPlaneDataObject.Enabled.GetValue();
  m_scenePlane->setVisible(m_clipPlaneDataObject.ShowHelper.GetValue());
  m_scenePlane->m_transform.m_center = m_clipPlaneDataObject.Position.GetValue();
  m_scenePlane->m_transform.m_rotation = m_clipPlaneDataObject.Rotation.GetValue();
  m_scenePlane->updateTransform();

  // Notify observers
  for (auto& observer : m_scenePlane->m_observers) {
    observer(plane);
  }

  // Call dirty callback if set
  if (m_dirtyCallback) {
    m_dirtyCallback();
  }
}

void
ClipPlaneObject::EnabledChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateObjectFromProps();
}

void
ClipPlaneObject::ShowHelperChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateObjectFromProps();
}

void
ClipPlaneObject::PositionChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateObjectFromProps();
}

void
ClipPlaneObject::RotationChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateObjectFromProps();
}

void
ClipPlaneObject::fromDocument(docReader* reader)
{
  // Peek at metadata
  uint32_t version = reader->peekVersion();
  std::string type = reader->peekObjectType();
  std::string name = reader->peekObjectName();

  reader->readPrty(&m_clipPlaneDataObject.Enabled);
  reader->readPrty(&m_clipPlaneDataObject.ShowHelper);
  reader->readPrty(&m_clipPlaneDataObject.Position);
  reader->readPrty(&m_clipPlaneDataObject.Rotation);
}

void
ClipPlaneObject::toDocument(docWriter* writer)
{
  writer->beginObject("clipPlane0", "ClipPlaneObject", ClipPlaneObject::CURRENT_VERSION);

  m_clipPlaneDataObject.Enabled.Write(*writer);
  m_clipPlaneDataObject.ShowHelper.Write(*writer);
  m_clipPlaneDataObject.Position.Write(*writer);
  m_clipPlaneDataObject.Rotation.Write(*writer);
  writer->endObject();
}
