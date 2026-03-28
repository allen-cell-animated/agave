#pragma once

#include <QObject>

class RenderSettings;
class SceneObject;

class QRenderSettings : public QObject
{
  Q_OBJECT

public:
  QRenderSettings(QObject* pParent = NULL);
  QRenderSettings(const QRenderSettings& Other);
  QRenderSettings& operator=(const QRenderSettings& Other);

  float GetDensityScale() const;
  void SetDensityScale(const float& DensityScale);
  int GetShadingType() const;
  void SetShadingType(const int& ShadingType);
  int GetRendererType() const;
  void SetRendererType(const int& RendererType);
  float GetGradientFactor() const;
  void SetGradientFactor(const float& GradientFactor);

  void setRenderSettings(RenderSettings& rs);
  RenderSettings* renderSettings() { return m_renderSettings; }

signals:
  void Changed();
  void ChangedRenderer(int);
  void Selected(SceneObject*);

private:
  int m_RendererType;
  RenderSettings* m_renderSettings;
};
