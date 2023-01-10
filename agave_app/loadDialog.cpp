#include "loadDialog.h"

#include "RangeWidget.h"

#include "renderlib/Logging.h"

#include <QDialogButtonBox>
#include <QLabel>
#include <QSpinBox>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QWidget>

LoadDialog::LoadDialog(std::string path, const std::vector<MultiscaleDims>& dims, QWidget* parent)
  : QDialog(parent)
{
  mPath = path;
  mDims = dims;
  mSelectedLevel = 0;
  mScene = 0;

  setWindowTitle(tr("Load Settings"));

  QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

  connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

  mSceneInput = new QSpinBox(this);
  mSceneInput->setMinimum(0);
  mSceneInput->setMaximum(65536);
  mSceneInput->setValue(0);

  mMetadataTree = new QTreeWidget(this);
  mMetadataTree->setColumnCount(2);
  mMetadataTree->setHeaderLabels(QStringList() << "Key"
                                               << "Value");
  mMetadataTree->setSelectionMode(QAbstractItemView::SingleSelection);
  std::string dimstring = "TCZYX";
  for (auto d : dims) {
    QTreeWidgetItem* item = new QTreeWidgetItem(QStringList() << "Level" << QString::fromStdString(d.path));
    item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
    item->setData(0, Qt::UserRole, QVariant::fromValue(QString::fromStdString(d.path)));
    for (size_t i = 0; i < d.shape.size(); ++i) {
      QTreeWidgetItem* child =
        new QTreeWidgetItem(QStringList() << QString::fromStdString(dimstring.substr(i, 1))
                                          << QString::number(d.shape[i]) + " (" + QString::number(d.scale[i]) + ")");
      child->setFlags(Qt::ItemIsEnabled);
      item->addChild(child);
    }

    mMetadataTree->addTopLevelItem(item);
  }
  connect(mMetadataTree, SIGNAL(itemSelectionChanged()), this, SLOT(onItemSelectionChanged()));

  connect(mSceneInput, SIGNAL(valueChanged(int)), this, SLOT(updateScene(int)));

  m_roiX = new RangeWidget(Qt::Horizontal);
  m_roiX->setStatusTip(tr("Region to load: X axis"));
  m_roiX->setToolTip(tr("Region to load: X axis"));
  m_roiX->setRange(0, dims[mSelectedLevel].shape[4]);
  m_roiX->setFirstValue(0);
  m_roiX->setSecondValue(m_roiX->maximum());
  QObject::connect(m_roiX, &RangeWidget::firstValueChanged, this, &LoadDialog::updateMemoryEstimate);
  QObject::connect(m_roiX, &RangeWidget::secondValueChanged, this, &LoadDialog::updateMemoryEstimate);
  m_roiY = new RangeWidget(Qt::Horizontal);
  m_roiY->setStatusTip(tr("Region to load: Y axis"));
  m_roiY->setToolTip(tr("Region to load: Y axis"));
  m_roiY->setRange(0, dims[mSelectedLevel].shape[3]);
  m_roiY->setFirstValue(0);
  m_roiY->setSecondValue(m_roiY->maximum());
  QObject::connect(m_roiY, &RangeWidget::firstValueChanged, this, &LoadDialog::updateMemoryEstimate);
  QObject::connect(m_roiY, &RangeWidget::secondValueChanged, this, &LoadDialog::updateMemoryEstimate);
  m_roiZ = new RangeWidget(Qt::Horizontal);
  m_roiZ->setStatusTip(tr("Region to load: Z axis"));
  m_roiZ->setToolTip(tr("Region to load: Z axis"));
  m_roiZ->setRange(0, dims[mSelectedLevel].shape[2]);
  m_roiZ->setFirstValue(0);
  m_roiZ->setSecondValue(m_roiZ->maximum());
  QObject::connect(m_roiZ, &RangeWidget::firstValueChanged, this, &LoadDialog::updateMemoryEstimate);
  QObject::connect(m_roiZ, &RangeWidget::secondValueChanged, this, &LoadDialog::updateMemoryEstimate);

  mMemoryEstimateLabel = new QLabel("Memory Estimate: 0 MB");
  updateMemoryEstimate();

  QVBoxLayout* layout = new QVBoxLayout(this);

  layout->addWidget(mSceneInput);
  layout->addWidget(mMetadataTree);
  layout->addWidget(m_roiX);
  layout->addWidget(m_roiY);
  layout->addWidget(m_roiZ);
  layout->addWidget(mMemoryEstimateLabel);
  layout->addWidget(buttonBox);
  setLayout(layout);
}

void
LoadDialog::updateScene(int value)
{
  // mSettings.mSceneIndex = value;
}

void
LoadDialog::updateMemoryEstimate()
{
  LoadSpec spec;
  spec.minx = m_roiX->firstValue();
  spec.maxx = m_roiX->secondValue();
  spec.miny = m_roiY->firstValue();
  spec.maxy = m_roiY->secondValue();
  spec.minz = m_roiZ->firstValue();
  spec.maxz = m_roiZ->secondValue();
  size_t mem = spec.getMemoryEstimate();
  std::string label = LoadSpec::bytesToStringLabel(mem);

  mMemoryEstimateLabel->setText("Memory Estimate: " + QString::fromStdString(label));
}

void
LoadDialog::onItemSelectionChanged()
{
  auto items = mMetadataTree->selectedItems();
  if (items.size() == 1) {
    auto item = items[0];
    if (item->parent() == nullptr) {
      // top level item
      auto level = item->text(1).toInt();
      float pct0x = m_roiX->firstPercent();
      float pct1x = m_roiX->secondPercent();
      float pct0y = m_roiY->firstPercent();
      float pct1y = m_roiY->secondPercent();
      float pct0z = m_roiZ->firstPercent();
      float pct1z = m_roiZ->secondPercent();
      m_roiX->setRange(0, mDims[level].shape[4]);
      m_roiX->setFirstValue(0 + pct0x * m_roiX->range());
      m_roiX->setSecondValue(0 + pct1x * m_roiX->range());
      m_roiY->setRange(0, mDims[level].shape[3]);
      m_roiY->setFirstValue(0 + pct0y * m_roiY->range());
      m_roiY->setSecondValue(0 + pct1y * m_roiY->range());
      m_roiZ->setRange(0, mDims[level].shape[2]);
      m_roiZ->setFirstValue(0 + pct0z * m_roiZ->range());
      m_roiZ->setSecondValue(0 + pct1z * m_roiZ->range());
      updateMemoryEstimate();
    }
  } else {
    LOG_ERROR << "Unexpected number of selected items: " << items.size();
  }
}

LoadSpec
LoadDialog::getLoadSpec() const
{
  LoadSpec spec;
  spec.filepath = mPath;
  spec.scene = mScene;
  spec.time = 0;
  spec.maxx = m_roiX->secondValue();
  spec.minx = m_roiX->firstValue();
  spec.maxy = m_roiY->secondValue();
  spec.miny = m_roiY->firstValue();
  spec.maxz = m_roiZ->secondValue();
  spec.minz = m_roiZ->firstValue();

  auto items = mMetadataTree->selectedItems();
  if (items.size() == 1) {
    auto item = items[0];
    if (item->parent() == nullptr) {
      // top level item
      auto level = item->text(1).toInt();
      spec.subpath = item->data(0, Qt::UserRole).toString().toStdString();
    }
  }
  return spec;
}