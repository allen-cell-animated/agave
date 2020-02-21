#pragma once

#include "Controls.h"
#include "renderlib/GradientData.h"

#include <QFormLayout>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>

#include <memory>

class QRenderSettings;
class ImageXYZC;
class RangeWidget;
class RenderSettings;
class Scene;
class Section;

class QAppearanceSettingsWidget : public QGroupBox
{
  Q_OBJECT

public:
  QAppearanceSettingsWidget(QWidget* pParent = NULL, QRenderSettings* qrs = nullptr, RenderSettings* rs = nullptr);

  void onNewImage(Scene* scene, std::string filepath);

public slots:
  void OnRenderBegin(void);
  void OnSetDensityScale(double DensityScale);
  void OnTransferFunctionChanged(void);
  void OnSetRendererType(int Index);
  void OnSetShadingType(int Index);
  void OnSetGradientFactor(double GradientFactor);
  void OnSetStepSizePrimaryRay(const double& StepSizePrimaryRay);
  void OnSetStepSizeSecondaryRay(const double& StepSizeSecondaryRay);

public:
  void OnBackgroundColorChanged(const QColor& color);
  void OnDiffuseColorChanged(int i, const QColor& color);
  void OnSpecularColorChanged(int i, const QColor& color);
  void OnEmissiveColorChanged(int i, const QColor& color);
  void OnUpdateLut(int i, const std::vector<LutControlPoint>& stops);

  void OnOpacityChanged(int i, double opacity);
  void OnRoughnessChanged(int i, double roughness);
  void OnChannelChecked(int i, bool is_checked);

  void OnSetAreaLightTheta(double value);
  void OnSetAreaLightPhi(double value);
  void OnSetAreaLightSize(double value);
  void OnSetAreaLightDistance(double value);
  void OnSetAreaLightColor(double intensity, const QColor& color);
  void OnSetSkyLightTopColor(double intensity, const QColor& color);
  void OnSetSkyLightMidColor(double intensity, const QColor& color);
  void OnSetSkyLightBotColor(double intensity, const QColor& color);

  void OnSetRoiXMax(int value);
  void OnSetRoiYMax(int value);
  void OnSetRoiZMax(int value);
  void OnSetRoiXMin(int value);
  void OnSetRoiYMin(int value);
  void OnSetRoiZMin(int value);

  void OnSetScaleX(double value);
  void OnSetScaleY(double value);
  void OnSetScaleZ(double value);

private:
  Scene* m_scene;
  std::string m_filepath;

  QFormLayout m_MainLayout;
  QNumericSlider m_DensityScaleSlider;
  QComboBox m_RendererType;
  QComboBox m_ShadingType;
  QNumericSlider m_GradientFactorSlider;
  QNumericSlider m_StepSizePrimaryRaySlider;
  QNumericSlider m_StepSizeSecondaryRaySlider;
  QColorPushButton m_backgroundColorButton;

  QRenderSettings* m_qrendersettings;

  Section* m_clipRoiSection;
  RangeWidget* m_roiX;
  RangeWidget* m_roiY;
  RangeWidget* m_roiZ;

  Section* m_scaleSection;
  QDoubleSpinner* m_xscaleSpinner;
  QDoubleSpinner* m_yscaleSpinner;
  QDoubleSpinner* m_zscaleSpinner;

  std::vector<Section*> m_channelSections;

  struct lt0
  {
    QNumericSlider* m_thetaSlider;
    QNumericSlider* m_phiSlider;
    QNumericSlider* m_sizeSlider;
    QNumericSlider* m_distSlider;
    QNumericSlider* m_intensitySlider;
    QColorPushButton* m_areaLightColorButton;
  } m_lt0gui;

  struct lt1
  {
    QNumericSlider* m_stintensitySlider;
    QColorPushButton* m_stColorButton;
    QNumericSlider* m_smintensitySlider;
    QColorPushButton* m_smColorButton;
    QNumericSlider* m_sbintensitySlider;
    QColorPushButton* m_sbColorButton;
  } m_lt1gui;

  Section* createLightingControls();
  void initLightingControls(Scene* scene);
};
