#include "loadDialog.h"

#include "Controls.h"
#include "RangeWidget.h"
#include "Section.h"

#include "renderlib/FileReader.h"
#include "renderlib/Logging.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QLabel>
#include <QListView>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QStyledItemDelegate>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QWidget>

LoadDialog::LoadDialog(std::string path, const std::vector<MultiscaleDims>& dims, uint32_t scene, QWidget* parent)
  : QDialog(parent)
{
  m_reader = FileReader::getReader(path);
  mPath = path;
  mDims = dims;
  mSelectedLevel = 0;
  mScene = scene;

  setWindowTitle(tr("Load Settings"));

  // get standard QLabel font size
  QFont f = QLabel("A").font();
  int standardPointSize = f.pointSize();

  // QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  // buttonBox->setCenterButtons(true);
  // buttonBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  // connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  // connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

  QPushButton* cancelButton = new QPushButton(tr("Cancel"));
  QPushButton* openButton = new QPushButton(tr("Open"));
  QHBoxLayout* buttonBox = new QHBoxLayout();
  buttonBox->addWidget(cancelButton);
  buttonBox->addWidget(openButton);
  connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);
  connect(openButton, &QPushButton::clicked, this, &QDialog::accept);

  mSceneInput = new QSpinBox(this);
  mSceneInput->setMinimum(0);
  mSceneInput->setMaximum(65536);
  mSceneInput->setValue(scene);
  mSceneInput->setVisible(false);

  mMultiresolutionInput = new QComboBox(this);
  updateMultiresolutionInput();

  m_TimeSlider = new QIntSlider(this);
  m_TimeSlider->setRange(0, 0);
  m_TimeSlider->setValue(0);
  int maxt = dims[0].shape[0];
  if (maxt > 1) {
    m_TimeSlider->setRange(0, maxt - 1);
  }
  m_TimeSlider->setEnabled(maxt > 1);

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
  mMetadataTree->setItemSelected(mMetadataTree->topLevelItem(0), true);

  mChannels = new QCheckList("Channels", this);
  mChannels->setTitleText("");
  for (int i = 0; i < dims[0].shape[1]; ++i) {
    std::string channelName = "Ch " + std::to_string(i);
    if (dims[0].channelNames.size() > i) {
      channelName = dims[0].channelNames[i];
    }
    mChannels->addCheckItem(QString::fromStdString(channelName), i, Qt::Checked);
  }

  connect(mMetadataTree, SIGNAL(itemSelectionChanged()), this, SLOT(onItemSelectionChanged()));
  // connect(mSceneInput, SIGNAL(valueChanged(int)), this, SLOT(updateScene(int)));
  connect(mMultiresolutionInput, SIGNAL(currentIndexChanged(int)), this, SLOT(updateMultiresolutionLevel(int)));
  connect(mChannels, SIGNAL(globalCheckStateChanged(int)), this, SLOT(updateChannels(int)));

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

  QVBoxLayout* roiLayout = new QVBoxLayout();
  roiLayout->addWidget(new QLabel("X"));
  roiLayout->addWidget(m_roiX);
  roiLayout->addWidget(new QLabel("Y"));
  roiLayout->addWidget(m_roiY);
  roiLayout->addWidget(new QLabel("Z"));
  roiLayout->addWidget(m_roiZ);

  m_roiSection = new Section("Region of Interest", 0);
  m_roiSection->setContentLayout(*roiLayout);
  m_roiSection->setEnabled(m_reader->supportChunkedLoading());

  QObject::connect(m_roiSection, &Section::collapsed, [this]() { this->adjustSize(); });

  mVolumeLabel = new QLabel("XxYxZ pixels");
  mMemoryEstimateLabel = new QLabel("Memory Estimate: 0 MB");
  QFont font = mMemoryEstimateLabel->font();
  font.setPointSize(font.pointSize() * 1.5);
  mMemoryEstimateLabel->setFont(font);

  updateMultiresolutionLevel(mSelectedLevel);

  QFormLayout* layout = new QFormLayout(this);
  layout->setLabelAlignment(Qt::AlignLeft);
  layout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

  static const int spacing = 6;
  // layout->addRow("Scene", mSceneInput);
  layout->addRow("Resolution Level", mMultiresolutionInput);
  layout->addItem(new QSpacerItem(0, spacing, QSizePolicy::Expanding, QSizePolicy::Expanding));
  layout->addRow("Time", m_TimeSlider);
  layout->addItem(new QSpacerItem(0, spacing, QSizePolicy::Expanding, QSizePolicy::Expanding));
  // layout->addWidget(mMetadataTree);
  mMetadataTree->hide();
  layout->addRow("Channels", mChannels);
  layout->addItem(new QSpacerItem(0, spacing, QSizePolicy::Expanding, QSizePolicy::Expanding));
  layout->addRow(m_roiSection);
  QFrame* hline = new QFrame();
  hline->setFrameShape(QFrame::HLine);
  layout->addRow(hline);
  layout->addItem(new QSpacerItem(0, spacing, QSizePolicy::Expanding, QSizePolicy::Expanding));
  layout->addRow(mVolumeLabel);
  layout->addRow(mMemoryEstimateLabel);
  layout->addItem(new QSpacerItem(0, spacing, QSizePolicy::Expanding, QSizePolicy::Expanding));
  layout->addRow(buttonBox);

  setLayout(layout);
}

LoadDialog::~LoadDialog()
{
  delete m_reader;
}

void
LoadDialog::updateScene(int value)
{
  mScene = value;
  mDims = m_reader->loadMultiscaleDims(mPath, mScene);
  updateMultiresolutionInput();
}

void
LoadDialog::updateChannels(int state)
{
  std::vector<uint32_t> channels = mChannels->getCheckedIndices();
  updateMemoryEstimate();
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

  mVolumeLabel->setText(QString::number(spec.maxx - spec.minx) + " x " + QString::number(spec.maxy - spec.miny) +
                        " x " + QString::number(spec.maxz - spec.minz) + " pixels");

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
      updateMultiresolutionLevel(level);
    }
  } else {
    LOG_ERROR << "Unexpected number of selected items: " << items.size();
  }
}

void
LoadDialog::updateMultiresolutionLevel(int level)
{
  mSelectedLevel = level;
  MultiscaleDims d = mDims[level];

  // update the time slider

  int t = m_TimeSlider->value();
  // maintain a t value at same percentage of total.
  float pct = (float)t / (float)m_TimeSlider->maximum();
  int maxt = d.shape[0];
  m_TimeSlider->setRange(0, maxt - 1);
  m_TimeSlider->setValue(pct * (maxt - 1));
  m_TimeSlider->setEnabled(maxt > 1);

  // update the channels
  mChannels->clear();
  for (int i = 0; i < d.shape[1]; ++i) {
    std::string channelName = "Ch " + std::to_string(i);
    if (d.channelNames.size() > i) {
      channelName = d.channelNames[i];
    }
    mChannels->addCheckItem(QString::fromStdString(channelName), i, Qt::Checked);
  }

  // update the xyz sliders

  float pct0x = m_roiX->firstPercent();
  float pct1x = m_roiX->secondPercent();
  float pct0y = m_roiY->firstPercent();
  float pct1y = m_roiY->secondPercent();
  float pct0z = m_roiZ->firstPercent();
  float pct1z = m_roiZ->secondPercent();
  m_roiX->setRange(0, d.shape[4]);
  m_roiX->setFirstValue(0 + pct0x * m_roiX->range());
  m_roiX->setSecondValue(0 + pct1x * m_roiX->range());
  m_roiY->setRange(0, d.shape[3]);
  m_roiY->setFirstValue(0 + pct0y * m_roiY->range());
  m_roiY->setSecondValue(0 + pct1y * m_roiY->range());
  m_roiZ->setRange(0, d.shape[2]);
  m_roiZ->setFirstValue(0 + pct0z * m_roiZ->range());
  m_roiZ->setSecondValue(0 + pct1z * m_roiZ->range());

  updateMemoryEstimate();
}

LoadSpec
LoadDialog::getLoadSpec() const
{
  LoadSpec spec;
  spec.filepath = mPath;
  spec.scene = mScene;
  spec.time = m_TimeSlider->value();
  spec.channels = mChannels->getCheckedIndices();

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

void
LoadDialog::updateMultiresolutionInput()
{
  mMultiresolutionInput->clear();
  for (auto d : mDims) {
    LoadSpec spec;
    spec.maxx = d.shape[4];
    spec.maxy = d.shape[3];
    spec.maxz = d.shape[2];
    size_t mem = spec.getMemoryEstimate();
    std::string label = LoadSpec::bytesToStringLabel(mem);

    mMultiresolutionInput->addItem(QString::fromStdString(d.path + " (" + label + " max.)"));
  }
  mMultiresolutionInput->setEnabled(mDims.size() > 1);
}