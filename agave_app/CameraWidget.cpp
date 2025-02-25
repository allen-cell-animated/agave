#include "CameraWidget.h"
#include "RenderSettings.h"

#include <QLabel>

struct GenericUIInfo
{
  std::string type;
  std::string formLabel;
  std::string statusTip;
  std::string toolTip;

  GenericUIInfo() = default;
  GenericUIInfo(std::string type, std::string formLabel, std::string statusTip, std::string toolTip)
    : type(type)
    , formLabel(formLabel)
    , statusTip(statusTip)
    , toolTip(toolTip)
  {
  }
};
struct CheckBoxUiInfo : public GenericUIInfo
{
  static constexpr const char* TYPE = "CheckBox";

  CheckBoxUiInfo() { type = CheckBoxUiInfo::TYPE; }
  CheckBoxUiInfo(std::string formLabel, std::string statusTip, std::string toolTip)
    : GenericUIInfo(CheckBoxUiInfo::TYPE, formLabel, statusTip, toolTip)
  {
  }
};
struct ComboBoxUiInfo : public GenericUIInfo
{
  static constexpr const char* TYPE = "ComboBox";
  std::vector<std::string> items;

  ComboBoxUiInfo() { type = ComboBoxUiInfo::TYPE; }
  ComboBoxUiInfo(std::string formLabel, std::string statusTip, std::string toolTip, std::vector<std::string> items)
    : GenericUIInfo(ComboBoxUiInfo::TYPE, formLabel, statusTip, toolTip)
    , items(items)
  {
  }
};
struct FloatSliderSpinnerUiInfo : public GenericUIInfo
{
  static constexpr const char* TYPE = "FloatSliderSpinner";
  float min = 0.0f;
  float max = 0.0f;
  int decimals = 0;
  float singleStep = 0.0f;
  int numTickMarks = 0;
  std::string suffix;

  FloatSliderSpinnerUiInfo() { type = FloatSliderSpinnerUiInfo::TYPE; }
  FloatSliderSpinnerUiInfo(std::string formLabel,
                           std::string statusTip,
                           std::string toolTip,
                           float min,
                           float max,
                           int decimals,
                           float singleStep,
                           int numTickMarks = 0,
                           std::string suffix = "")
    : GenericUIInfo(FloatSliderSpinnerUiInfo::TYPE, formLabel, statusTip, toolTip)
    , min(min)
    , max(max)
    , decimals(decimals)
    , singleStep(singleStep)
    , numTickMarks(numTickMarks)
    , suffix(suffix)
  {
  }
};
struct IntSliderSpinnerUiInfo : public GenericUIInfo
{
  static constexpr const char* TYPE = "IntSliderSpinner";
  int min;
  int max;
  int singleStep;
  int numTickMarks;
  std::string suffix;

  IntSliderSpinnerUiInfo() { type = IntSliderSpinnerUiInfo::TYPE; }
};

QNumericSlider*
create(const FloatSliderSpinnerUiInfo* info, std::shared_ptr<prtyProperty<float>> prop)
{
  QNumericSlider* slider = new QNumericSlider();
  slider->setStatusTip(QString::fromStdString(info->statusTip));
  slider->setToolTip(QString::fromStdString(info->toolTip));
  slider->setRange(info->min, info->max);
  slider->setDecimals(info->decimals);
  slider->setSingleStep(info->singleStep);
  slider->setNumTickMarks(info->numTickMarks);
  slider->setSuffix(QString::fromStdString(info->suffix));

  slider->setValue(prop->get(), true);
  QObject::connect(slider, &QNumericSlider::valueChanged, [slider, prop](double value) { prop->set(value, true); });
  // TODO how would this capture the "previous" value, for undo?
  QObject::connect(slider, &QNumericSlider::valueChangeCommit, [slider, prop]() { prop->notifyAll(true); });

  return slider;
}
QNumericSlider*
create(const IntSliderSpinnerUiInfo* info, std::shared_ptr<prtyProperty<int>> prop)
{
  QNumericSlider* slider = new QNumericSlider();
  slider->setStatusTip(QString::fromStdString(info->statusTip));
  slider->setToolTip(QString::fromStdString(info->toolTip));
  slider->setRange(info->min, info->max);
  slider->setSingleStep(info->singleStep);
  slider->setNumTickMarks(info->numTickMarks);
  slider->setSuffix(QString::fromStdString(info->suffix));

  slider->setValue(prop->get(), true);
  QObject::connect(slider, &QNumericSlider::valueChanged, [slider, prop](double value) { prop->set(value, true); });
  // TODO how would this capture the "previous" value, for undo?
  QObject::connect(slider, &QNumericSlider::valueChangeCommit, [slider, prop]() { prop->notifyAll(true); });

  return slider;
}
QCheckBox*
create(const CheckBoxUiInfo* info, std::shared_ptr<prtyProperty<bool>> prop)
{
  QCheckBox* checkBox = new QCheckBox();
  checkBox->setStatusTip(QString::fromStdString(info->statusTip));
  checkBox->setToolTip(QString::fromStdString(info->toolTip));
  // checkBox->setText(QString::fromStdString(info->formLabel));
  checkBox->setCheckState(prop->get() ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
  QObject::connect(checkBox, &QCheckBox::stateChanged, [checkBox, prop](int state) {
    prop->set(state == Qt::CheckState::Checked, true);
  });
  return checkBox;
}
QComboBox*
create(const ComboBoxUiInfo* info, std::shared_ptr<prtyProperty<int>> prop)
{
  QComboBox* comboBox = new QComboBox();
  comboBox->setStatusTip(QString::fromStdString(info->statusTip));
  comboBox->setToolTip(QString::fromStdString(info->toolTip));
  for (const auto& item : info->items) {
    comboBox->addItem(QString::fromStdString(item));
  }
  comboBox->setCurrentIndex(prop->get());
  QObject::connect(comboBox, &QComboBox::currentIndexChanged, [comboBox, prop](int index) { prop->set(index, true); });
  return comboBox;
}

void
createSection(QFormLayout* layout, std::vector<GenericUIInfo*> controlDescs)
{
  // for (const auto& desc : controlDescs) {
  //   QLabel* label = new QLabel(QString::fromStdString(desc->formLabel));
  //   if (desc->type == CheckBoxUiInfo::TYPE) {
  //     layout->addRow(label, create(static_cast<const CheckBoxUiInfo*>(desc), ));
  //   } else if (desc->type == ComboBoxUiInfo::TYPE) {
  //     layout->addRow(label, create(static_cast<const ComboBoxUiInfo&>(desc)));
  //   } else if (desc->type == FloatSliderSpinnerUiInfo::TYPE) {
  //     layout->addRow(label, create(static_cast<const FloatSliderSpinnerUiInfo&>(desc)));
  //   } else if (desc->type == IntSliderSpinnerUiInfo::TYPE) {
  //     layout->addRow(label, create(static_cast<const IntSliderSpinnerUiInfo&>(desc)));
  //   }
  // }
}

QNumericSlider*
addRow(const FloatSliderSpinnerUiInfo& info, prtyProperty<float>* prop)
{
  QNumericSlider* slider = new QNumericSlider();
  slider->setStatusTip(QString::fromStdString(info.statusTip));
  slider->setToolTip(QString::fromStdString(info.toolTip));
  slider->setRange(info.min, info.max);
  slider->setDecimals(info.decimals);
  slider->setSingleStep(info.singleStep);
  slider->setNumTickMarks(info.numTickMarks);
  slider->setSuffix(QString::fromStdString(info.suffix));

  slider->setValue(prop->get(), true);
  QObject::connect(slider, &QNumericSlider::valueChanged, [slider, prop](double value) { prop->set(value, true); });
  // TODO how would this capture the "previous" value, for undo?
  QObject::connect(slider, &QNumericSlider::valueChangeCommit, [slider, prop]() { prop->notifyAll(true); });

  return slider;
}
QComboBox*
addRow(const ComboBoxUiInfo& info, prtyProperty<int>* prop)
{
  QComboBox* comboBox = new QComboBox();
  comboBox->setStatusTip(QString::fromStdString(info.statusTip));
  comboBox->setToolTip(QString::fromStdString(info.toolTip));
  for (const auto& item : info.items) {
    comboBox->addItem(QString::fromStdString(item));
  }
  comboBox->setCurrentIndex(prop->get());
  QObject::connect(comboBox, &QComboBox::currentIndexChanged, [comboBox, prop](int index) { prop->set(index, true); });
  return comboBox;
}
QCheckBox*
addRow(const CheckBoxUiInfo& info, prtyProperty<bool>* prop)
{
  QCheckBox* checkBox = new QCheckBox();
  checkBox->setStatusTip(QString::fromStdString(info.statusTip));
  checkBox->setToolTip(QString::fromStdString(info.toolTip));
  // checkBox->setText(QString::fromStdString(info.formLabel));
  checkBox->setCheckState(prop->get() ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
  QObject::connect(checkBox, &QCheckBox::stateChanged, [checkBox, prop](int state) {
    prop->set(state == Qt::CheckState::Checked, true);
  });
  return checkBox;
}

QCameraWidget::QCameraWidget(QWidget* pParent, QCamera* cam, RenderSettings* rs, CameraDataObject* cdo)
  : QWidget(pParent)
  , m_MainLayout()
  , m_qcamera(cam)
  , m_renderSettings(rs)
  , m_cameraDataObject(cdo)
{
  Controls::initFormLayout(m_MainLayout);
  setLayout(&m_MainLayout);

  QNumericSlider* slider = addRow(FloatSliderSpinnerUiInfo("Exposure",
                                                           "Set Exposure",
                                                           "Set camera exposure",
                                                           0.0f,
                                                           1.0f,
                                                           2,    // decimals
                                                           0.01, // singleStep
                                                           0     // numTickMarks
                                                           ),
                                  &m_cameraDataObject->Exposure);
  m_MainLayout.addRow("Exposure", slider);
  QComboBox* comboBox = addRow(ComboBoxUiInfo("Exposure Time",
                                              "Set Exposure Time",
                                              "Set number of samples to accumulate per viewport update",
                                              { "1", "2", "4", "8" }),
                               &m_cameraDataObject->ExposureIterations);
  m_MainLayout.addRow("Exposure Time", comboBox);
  QCheckBox* checkBox = addRow(CheckBoxUiInfo("Noise Reduction", "Enable denoising pass", "Enable denoising pass"),
                               &m_cameraDataObject->NoiseReduction);
  m_MainLayout.addRow("Noise Reduction", checkBox);
  QNumericSlider* slider2 = addRow(FloatSliderSpinnerUiInfo("Aperture Size",
                                                            "Set camera aperture size",
                                                            "Set camera aperture size",
                                                            0.0f,
                                                            0.1f,
                                                            2,    // decimals
                                                            0.01, // singleStep
                                                            0,    // numTickMarks
                                                            " mm"),
                                   &m_cameraDataObject->ApertureSize);
  m_MainLayout.addRow("Aperture Size", slider2);
  QNumericSlider* slider3 = addRow(FloatSliderSpinnerUiInfo("Field of view",
                                                            "Set camera field of view angle",
                                                            "Set camera field of view angle",
                                                            10.0f,
                                                            150.0f,
                                                            2,    // decimals
                                                            0.01, // singleStep
                                                            0,    // numTickMarks
                                                            " deg."),
                                   &m_cameraDataObject->FieldOfView);
  m_MainLayout.addRow("Field of view", slider3);
  QNumericSlider* slider4 = addRow(FloatSliderSpinnerUiInfo("Focal distance",
                                                            "Set focal distance",
                                                            "Set focal distance",
                                                            0.0f,
                                                            15.0f,
                                                            2,    // decimals
                                                            0.01, // singleStep
                                                            0,    // numTickMarks
                                                            " m"),
                                   &m_cameraDataObject->FocalDistance);
  m_MainLayout.addRow("Focal distance", slider4);

#if 0  
  // Exposure, controls how bright or dim overall scene is
  m_ExposureSlider.setStatusTip(tr("Set Exposure"));
  m_ExposureSlider.setToolTip(tr("Set camera exposure"));
  m_ExposureSlider.setRange(0.0f, 1.0f);
  m_ExposureSlider.setValue(cam->GetFilm().GetExposure());
  m_ExposureSlider.setDecimals(2);
  m_ExposureSlider.setSingleStep(0.01);

  m_MainLayout.addRow("Exposure", &m_ExposureSlider);

  connect(&m_ExposureSlider, &QNumericSlider::valueChanged, this, &QCameraWidget::SetExposure);

  // Number of render iterations per viewport update
  m_ExposureIterationsSpinner.setStatusTip(tr("Set Exposure Time"));
  m_ExposureIterationsSpinner.setToolTip(tr("Set number of samples to accumulate per viewport update"));
  m_ExposureIterationsSpinner.addItem("1", 1);
  m_ExposureIterationsSpinner.addItem("2", 2);
  m_ExposureIterationsSpinner.addItem("4", 4);
  m_ExposureIterationsSpinner.addItem("8", 8);
  m_ExposureIterationsSpinner.setCurrentIndex(
    m_ExposureIterationsSpinner.findData(cam->GetFilm().GetExposureIterations()));
  m_MainLayout.addRow("Exposure Time", &m_ExposureIterationsSpinner);
  connect(&m_ExposureIterationsSpinner, &QComboBox::currentIndexChanged, this, &QCameraWidget::SetExposureIterations);

  m_NoiseReduction.setStatusTip(tr("Enable denoising pass"));
  m_NoiseReduction.setToolTip(tr("Enable denoising pass"));
  m_NoiseReduction.setCheckState(rs->m_DenoiseParams.m_Enabled ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
  m_MainLayout.addRow("Noise Reduction", &m_NoiseReduction);

  connect(&m_NoiseReduction, &QCheckBox::stateChanged, this, &QCameraWidget::OnNoiseReduction);

  m_ApertureSizeSlider.setStatusTip(tr("Set camera aperture size"));
  m_ApertureSizeSlider.setToolTip(tr("Set camera aperture size"));
  m_ApertureSizeSlider.setRange(0.0, 0.1);
  m_ApertureSizeSlider.setSuffix(" mm");
  m_ApertureSizeSlider.setDecimals(2);
  m_ApertureSizeSlider.setValue(0.0);
  m_ApertureSizeSlider.setSingleStep(0.01);
  m_MainLayout.addRow("Aperture Size", &m_ApertureSizeSlider);

  connect(&m_ApertureSizeSlider, &QNumericSlider::valueChanged, this, &QCameraWidget::SetAperture);

  m_FieldOfViewSlider.setStatusTip(tr("Set camera field of view angle"));
  m_FieldOfViewSlider.setToolTip(tr("Set camera field of view angle"));
  m_FieldOfViewSlider.setRange(10.0, 150.0);
  m_FieldOfViewSlider.setDecimals(2);
  m_FieldOfViewSlider.setValue(cam->GetProjection().GetFieldOfView());
  m_FieldOfViewSlider.setSuffix(" deg.");
  m_MainLayout.addRow("Field of view", &m_FieldOfViewSlider);

  connect(&m_FieldOfViewSlider, &QNumericSlider::valueChanged, this, &QCameraWidget::SetFieldOfView);

  // Focal distance
  m_FocalDistanceSlider.setStatusTip(tr("Set focal distance"));
  m_FocalDistanceSlider.setToolTip(tr("Set focal distance"));
  m_FocalDistanceSlider.setRange(0.0, 15.0);
  m_FocalDistanceSlider.setDecimals(2);
  m_FocalDistanceSlider.setValue(0.0);
  m_FocalDistanceSlider.setSuffix(" m");

  m_MainLayout.addRow("Focal distance", &m_FocalDistanceSlider);

  connect(&m_FocalDistanceSlider, &QNumericSlider::valueChanged, this, &QCameraWidget::SetFocalDistance);

  QObject::connect(&cam->GetFilm(), SIGNAL(Changed(const QFilm&)), this, SLOT(OnFilmChanged()));
  QObject::connect(&cam->GetAperture(), SIGNAL(Changed(const QAperture&)), this, SLOT(OnApertureChanged()));
  QObject::connect(&cam->GetProjection(), SIGNAL(Changed(const QProjection&)), this, SLOT(OnProjectionChanged()));
  QObject::connect(&cam->GetFocus(), SIGNAL(Changed(const QFocus&)), this, SLOT(OnFocusChanged()));
#endif
}

void
QCameraWidget::OnFilmChanged()
{
  m_ExposureSlider.setValue(m_qcamera->GetFilm().GetExposure(), true);
  m_ExposureIterationsSpinner.blockSignals(true);
  m_ExposureIterationsSpinner.setCurrentIndex(
    m_ExposureIterationsSpinner.findData(m_qcamera->GetFilm().GetExposureIterations()));
  m_ExposureIterationsSpinner.blockSignals(false);
  m_NoiseReduction.blockSignals(true);
  m_NoiseReduction.setCheckState(m_renderSettings->m_DenoiseParams.m_Enabled ? Qt::CheckState::Checked
                                                                             : Qt::CheckState::Unchecked);
  m_NoiseReduction.blockSignals(false);
  emit m_qcamera->Changed();
}
void
QCameraWidget::OnApertureChanged()
{
  m_ApertureSizeSlider.setValue(m_qcamera->GetAperture().GetSize(), true);
  emit m_qcamera->Changed();
}
void
QCameraWidget::OnProjectionChanged()
{
  m_FieldOfViewSlider.setValue(m_qcamera->GetProjection().GetFieldOfView(), true);
  emit m_qcamera->Changed();
}
void
QCameraWidget::OnFocusChanged()
{
  m_FocalDistanceSlider.setValue(m_qcamera->GetFocus().GetFocalDistance(), true);
  emit m_qcamera->Changed();
}

QSize
QCameraWidget::sizeHint() const
{
  return QSize(20, 20);
}

void
QCameraWidget::SetExposure(const double& Exposure)
{
  m_qcamera->GetFilm().SetExposure(Exposure);
}

void
QCameraWidget::SetExposureIterations(int index)
{
  int value = m_ExposureIterationsSpinner.currentData().toInt();
  m_qcamera->GetFilm().SetExposureIterations(value);
}

void
QCameraWidget::OnNoiseReduction(const int& ReduceNoise)
{
  m_qcamera->GetFilm().SetNoiseReduction(m_NoiseReduction.checkState());
}

void
QCameraWidget::SetAperture(const double& Aperture)
{
  m_qcamera->GetAperture().SetSize(Aperture);
}

void
QCameraWidget::SetFieldOfView(const double& FieldOfView)
{
  m_qcamera->GetProjection().SetFieldOfView(FieldOfView);
}

void
QCameraWidget::SetFocalDistance(const double& FocalDistance)
{
  m_qcamera->GetFocus().SetFocalDistance(FocalDistance);
}
