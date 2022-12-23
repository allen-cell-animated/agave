#include "loadDialog.h"

#include "RangeWidget.h"

#include "renderlib/VolumeDimensions.h"

#include <QLabel>
#include <QSpinBox>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QWidget>

LoadDialog::LoadDialog(std::string path, const std::vector<MultiscaleDims>& dims, QWidget* parent)
  : QDialog(parent)
{
  setWindowTitle(tr("Load Settings"));

  mSceneInput = new QSpinBox(this);
  mSceneInput->setMinimum(0);
  mSceneInput->setMaximum(65536);
  mSceneInput->setValue(0);

  // struct MultiscaleDims
  // {
  //   std::vector<float> scale;
  //   std::vector<int64_t> shape;
  //   std::string dtype;
  //   std::string path;
  // };
  // MultiscaleDims dims[] = { { { 1, 1, 1 }, { 100, 100, 100 }, "tensorstore::DataType::uint8", "0" },
  //                           { { 2, 2, 2 }, { 50, 50, 50 }, "tensorstore::DataType::uint8", "1" },
  //                           { { 4, 4, 4 }, { 25, 25, 25 }, "tensorstore::DataType::uint8", "2" } };

  mMetadataTree = new QTreeWidget(this);
  mMetadataTree->setColumnCount(2);
  mMetadataTree->setHeaderLabels(QStringList() << "Key"
                                               << "Value");
  std::string dimstring = "TCZYX";
  for (auto d : dims) {
    QTreeWidgetItem* item = new QTreeWidgetItem(QStringList() << "Level" << QString::fromStdString(d.path));
    for (size_t i = 0; i < d.shape.size(); ++i) {
      item->addChild(new QTreeWidgetItem(QStringList()
                                         << QString::fromStdString(dimstring.substr(i, 1))
                                         << QString::number(d.shape[i]) + " (" + QString::number(d.scale[i]) + ")"));
    }

    // item->addChild(new QTreeWidgetItem(QStringList()
    //                                    << "Shape"
    //                                    << (QStringList() << QString::number(d.shape[0]) <<
    //                                    QString::number(d.shape[1])
    //                                                      << QString::number(d.shape[2]))
    //                                         .join(",")));
    // item->addChild(new QTreeWidgetItem(QStringList() << "DType" << QString::fromStdString(d.dtype)));
    mMetadataTree->addTopLevelItem(item);
  }

  connect(mSceneInput, SIGNAL(valueChanged(int)), this, SLOT(updateScene(int)));

  m_roiX = new RangeWidget(Qt::Horizontal);
  m_roiX->setStatusTip(tr("Region to load: X axis"));
  m_roiX->setToolTip(tr("Region to load: X axis"));
  m_roiX->setRange(0, 100);
  m_roiX->setFirstValue(0);
  m_roiX->setSecondValue(100);
  // QObject::connect(m_roiX, &RangeWidget::firstValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiXMin);
  // QObject::connect(m_roiX, &RangeWidget::secondValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiXMax);
  m_roiY = new RangeWidget(Qt::Horizontal);
  m_roiY->setStatusTip(tr("Region to load: Y axis"));
  m_roiY->setToolTip(tr("Region to load: Y axis"));
  m_roiY->setRange(0, 100);
  m_roiY->setFirstValue(0);
  m_roiY->setSecondValue(100);
  // QObject::connect(m_roiY, &RangeWidget::firstValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiYMin);
  // QObject::connect(m_roiY, &RangeWidget::secondValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiYMax);
  m_roiZ = new RangeWidget(Qt::Horizontal);
  m_roiZ->setStatusTip(tr("Region to load: Z axis"));
  m_roiZ->setToolTip(tr("Region to load: Z axis"));
  m_roiZ->setRange(0, 100);
  m_roiZ->setFirstValue(0);
  m_roiZ->setSecondValue(100);
  // QObject::connect(m_roiZ, &RangeWidget::firstValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiZMin);
  // QObject::connect(m_roiZ, &RangeWidget::secondValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiZMax);

  mMemoryEstimateLabel = new QLabel("Memory Estimate: 0 MB");

  QVBoxLayout* layout = new QVBoxLayout(this);

  layout->addWidget(mSceneInput);
  layout->addWidget(mMetadataTree);
  layout->addWidget(m_roiX);
  layout->addWidget(m_roiY);
  layout->addWidget(m_roiZ);
  layout->addWidget(mMemoryEstimateLabel);
  setLayout(layout);
}

void
LoadDialog::updateScene(int value)
{
  // mSettings.mSceneIndex = value;
}
