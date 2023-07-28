#include "loadDialog.h"

#include "Controls.h"
#include "RangeWidget.h"
#include "Section.h"

#include "renderlib/Logging.h"
#include "renderlib/io/FileReader.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QLabel>
#include <QListView>
#include <QListWidget>
#include <QMessageBox>
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
  setFocusPolicy(Qt::StrongFocus);

  // get standard QLabel font size
  QFont f = QLabel("A").font();
  int standardPointSize = f.pointSize();

  QPushButton* cancelButton = new QPushButton(tr("Cancel"));
  QPushButton* openButton = new QPushButton(tr("Open"));
  QHBoxLayout* buttonBox = new QHBoxLayout();
  buttonBox->addWidget(cancelButton);
  buttonBox->addWidget(openButton);
  connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);
  connect(openButton, &QPushButton::clicked, this, &LoadDialog::accept);

  mSceneInput = new QSpinBox(this);
  mSceneInput->setMinimum(0);
  mSceneInput->setMaximum(65536);
  mSceneInput->setValue(scene);
  mSceneInput->setVisible(false);

  mMultiresolutionInput = new QComboBox();
  updateMultiresolutionInput();

  m_TimeSlider = new QIntSlider();
  m_TimeSlider->setSpinnerKeyboardTracking(true);
  int maxt = dims[0].sizeT();
  if (maxt > 1) {
    m_TimeSlider->setRange(0, maxt - 1);
  } else {
    m_TimeSlider->setRange(0, 0);
  }
  m_TimeSlider->setValue(0);
  m_TimeSlider->setSingleStep(1);
  m_TimeSlider->setTickPosition(QSlider::TickPosition::TicksBelow);
  m_TimeSlider->setTickInterval((maxt - 1) / 10);
  m_TimeSlider->setEnabled(maxt > 1);

  mChannels = new QListWidget(this);
  mChannelsSection = new Section("Channels", 0);
  populateChannels(0);
  auto chseclo = new QVBoxLayout();
  chseclo->setContentsMargins(0, 0, 0, 0);
  chseclo->setSpacing(0);
  chseclo->addWidget(mChannels);

  // Make channels list resize at will
  mChannels->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
  mChannels->setMaximumHeight(mChannels->sizeHintForColumn(0)); // TODO: Add some margin

  chseclo->setSizeConstraint(QLayout::SetMinAndMaxSize);
  mChannels->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

  mChannelsSection->setContentLayout(*chseclo);
  mChannelsSection->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  QObject::connect(mChannelsSection, &Section::collapsed, [this]() { this->adjustSize(); });

  connect(mMultiresolutionInput, SIGNAL(currentIndexChanged(int)), this, SLOT(updateMultiresolutionLevel(int)));

  connect(mChannels, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(updateChannels()));

  m_roiX = new RangeWidget(Qt::Horizontal);
  m_roiX->setStatusTip(tr("Region to load: X axis"));
  m_roiX->setToolTip(tr("Region to load: X axis"));
  m_roiX->setRange(0, dims[mSelectedLevel].sizeX());
  m_roiX->setFirstValue(0);
  m_roiX->setSecondValue(m_roiX->maximum());
  QObject::connect(m_roiX, &RangeWidget::firstValueChanged, this, &LoadDialog::updateMemoryEstimate);
  QObject::connect(m_roiX, &RangeWidget::secondValueChanged, this, &LoadDialog::updateMemoryEstimate);
  m_roiY = new RangeWidget(Qt::Horizontal);
  m_roiY->setStatusTip(tr("Region to load: Y axis"));
  m_roiY->setToolTip(tr("Region to load: Y axis"));
  m_roiY->setRange(0, dims[mSelectedLevel].sizeY());
  m_roiY->setFirstValue(0);
  m_roiY->setSecondValue(m_roiY->maximum());
  QObject::connect(m_roiY, &RangeWidget::firstValueChanged, this, &LoadDialog::updateMemoryEstimate);
  QObject::connect(m_roiY, &RangeWidget::secondValueChanged, this, &LoadDialog::updateMemoryEstimate);
  m_roiZ = new RangeWidget(Qt::Horizontal);
  m_roiZ->setStatusTip(tr("Region to load: Z axis"));
  m_roiZ->setToolTip(tr("Region to load: Z axis"));
  m_roiZ->setRange(0, dims[mSelectedLevel].sizeZ());
  m_roiZ->setFirstValue(0);
  m_roiZ->setSecondValue(m_roiZ->maximum());
  QObject::connect(m_roiZ, &RangeWidget::firstValueChanged, this, &LoadDialog::updateMemoryEstimate);
  QObject::connect(m_roiZ, &RangeWidget::secondValueChanged, this, &LoadDialog::updateMemoryEstimate);

  QFormLayout* roiLayout = new QFormLayout();
  auto xlabel = new QLabel("X");
  xlabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  auto ylabel = new QLabel("Y");
  ylabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  auto zlabel = new QLabel("Z");
  zlabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  roiLayout->setLabelAlignment(Qt::AlignLeft);
  roiLayout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
  roiLayout->setSizeConstraint(QLayout::SetMinimumSize);
  roiLayout->addRow(xlabel, m_roiX);
  roiLayout->addRow(ylabel, m_roiY);
  roiLayout->addRow(zlabel, m_roiZ);

  m_roiSection = new Section("Region of Interest", 0);
  m_roiSection->setContentLayout(*roiLayout);
  m_roiSection->setEnabled(m_reader->supportChunkedLoading());

  QObject::connect(m_roiSection, &Section::collapsed, [this]() { this->adjustSize(); });

  mVolumeLabel = new QLabel("XxYxZ pixels");
  mMemoryEstimateLabel = new QLabel("Memory Estimate: 0 MB");
  QFont font = mMemoryEstimateLabel->font();
  font.setPointSize(font.pointSize() * 1.5);
  mMemoryEstimateLabel->setTextFormat(Qt::RichText);
  mMemoryEstimateLabel->setFont(font);

  updateMultiresolutionLevel(mSelectedLevel);

  QFormLayout* layout = new QFormLayout(this);
  layout->setLabelAlignment(Qt::AlignLeft);
  layout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

  static const int spacing = 4;
  // layout->addRow("Scene", mSceneInput);
  if (mMultiresolutionInput->count() > 1) {
    layout->addRow("Resolution Level", mMultiresolutionInput);
    layout->addItem(new QSpacerItem(0, spacing, QSizePolicy::Expanding, QSizePolicy::Minimum));
  }
  if (m_TimeSlider->isEnabled()) {
    layout->addRow("Time", m_TimeSlider);
    layout->addItem(new QSpacerItem(0, spacing, QSizePolicy::Expanding, QSizePolicy::Minimum));
  }
  layout->addRow(mChannelsSection);
  layout->addItem(new QSpacerItem(0, spacing, QSizePolicy::Expanding, QSizePolicy::Expanding));
  if (m_roiSection->isEnabled()) {
    layout->addRow(m_roiSection);
    layout->addItem(new QSpacerItem(0, spacing, QSizePolicy::Expanding, QSizePolicy::Expanding));
  }
  QFrame* hline = new QFrame();
  hline->setFrameShape(QFrame::HLine);
  layout->addRow(hline);
  layout->addItem(new QSpacerItem(0, spacing, QSizePolicy::Minimum, QSizePolicy::Minimum));
  layout->addRow(mVolumeLabel);
  layout->addItem(new QSpacerItem(0, spacing, QSizePolicy::Expanding, QSizePolicy::Minimum));
  layout->addRow(mMemoryEstimateLabel);
  layout->addItem(new QSpacerItem(0, spacing, QSizePolicy::Expanding, QSizePolicy::Minimum));
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
LoadDialog::updateChannels()
{
  std::vector<uint32_t> channels = getCheckedChannels();
  mChannelsSection->setTitle("Channels (" + QString::number(channels.size()) + "/" +
                             QString::number(mChannels->count()) + " selected)");
  updateMemoryEstimate();
}

std::vector<uint32_t>
LoadDialog::getCheckedChannels() const
{
  std::vector<uint32_t> channels;
  for (int i = 0; i < mChannels->count(); i++) {
    if (mChannels->item(i)->checkState() == Qt::Checked) {
      channels.push_back(i);
      // channels.push_back(mChannels->item(i)->data(Qt::UserRole).toUInt());
    }
  }
  return channels;
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

  mMemoryEstimateLabel->setText("Memory Estimate: <b>" + QString::fromStdString(label) + "</b>");
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
  int maxt = d.sizeT();
  m_TimeSlider->setRange(0, maxt - 1);
  m_TimeSlider->setValue(pct * (maxt - 1));
  m_TimeSlider->setEnabled(maxt > 1);

  // update the channels
  populateChannels(level);
  updateChannels();

  // update the xyz sliders

  float pct0x = m_roiX->firstPercent();
  float pct1x = m_roiX->secondPercent();
  float pct0y = m_roiY->firstPercent();
  float pct1y = m_roiY->secondPercent();
  float pct0z = m_roiZ->firstPercent();
  float pct1z = m_roiZ->secondPercent();
  m_roiX->setRange(0, d.sizeX());
  m_roiX->setFirstValue(0 + pct0x * m_roiX->range());
  m_roiX->setSecondValue(0 + pct1x * m_roiX->range());
  m_roiY->setRange(0, d.sizeY());
  m_roiY->setFirstValue(0 + pct0y * m_roiY->range());
  m_roiY->setSecondValue(0 + pct1y * m_roiY->range());
  m_roiZ->setRange(0, d.sizeZ());
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
  spec.channels = getCheckedChannels();

  spec.maxx = m_roiX->secondValue();
  spec.minx = m_roiX->firstValue();
  spec.maxy = m_roiY->secondValue();
  spec.miny = m_roiY->firstValue();
  spec.maxz = m_roiZ->secondValue();
  spec.minz = m_roiZ->firstValue();

  spec.subpath = mDims[mSelectedLevel].path;

  return spec;
}

void
LoadDialog::updateMultiresolutionInput()
{
  mMultiresolutionInput->clear();
  for (auto d : mDims) {
    LoadSpec spec;
    spec.maxx = d.sizeX();
    spec.maxy = d.sizeY();
    spec.maxz = d.sizeZ();
    size_t mem = spec.getMemoryEstimate();
    std::string label = LoadSpec::bytesToStringLabel(mem);

    mMultiresolutionInput->addItem(QString::fromStdString(d.path + " (" + label + " max.)"));
  }
  mMultiresolutionInput->setEnabled(mDims.size() > 1);
}

void
LoadDialog::accept()
{
  // validate inputs.
  std::vector<uint32_t> channels = getCheckedChannels();
  if (channels.size() == 0) {
    QMessageBox::warning(this, "No Channels Selected", "Please select at least one channel.");
    return;
  }

  QDialog::accept();
}

void
LoadDialog::populateChannels(int level)
{
  int nch = mDims[level].sizeC();
  int oldnch = mChannels->count();

  std::vector<uint32_t> channels = getCheckedChannels();

  mChannels->clear();
  for (int i = 0; i < nch; ++i) {
    std::string channelName = "Ch " + std::to_string(i);
    if (mDims[level].channelNames.size() > i) {
      channelName = mDims[level].channelNames[i];
    }
    QListWidgetItem* listItem = new QListWidgetItem(QString::fromStdString(channelName), mChannels);

    // if we are within the previous number of channels, then we can use the previous selection.
    if (i < oldnch) {
      if (std::find(channels.begin(), channels.end(), i) != channels.end()) {
        listItem->setCheckState(Qt::Checked);
      } else {
        listItem->setCheckState(Qt::Unchecked);
      }
    } else {
      // if we are at a greater value then include channel by default.
      listItem->setCheckState(Qt::Checked);
    }

    listItem->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
    listItem->setData(Qt::UserRole, QVariant::fromValue(i));
    mChannels->addItem(listItem);
  }
}