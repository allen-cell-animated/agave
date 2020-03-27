#include "TimelineDockWidget.h"

#include "Controls.h"
#include "QRenderSettings.h"

#include "renderlib/AppScene.h"
#include "renderlib/FileReader.h"
#include "renderlib/ImageXYZC.h"
#include "renderlib/Logging.h"
#include "renderlib/RenderSettings.h"

#include <QApplication>
#include <QScrollArea>

QTimelineWidget::QTimelineWidget(QWidget* pParent, QRenderSettings* qrs)
  : QWidget(pParent)
  , m_MainLayout()
  , m_qrendersettings(qrs)
  , m_scene(nullptr)
{
  // Create main layout
  m_MainLayout.setAlignment(Qt::AlignTop);
  setLayout(&m_MainLayout);

  QScrollArea* scrollArea = new QScrollArea();
  scrollArea->setWidgetResizable(true);

  auto* fullLayout = new QVBoxLayout();

  m_TimeSlider = new QIntSlider();
  m_TimeSlider->setStatusTip(tr("Set current time sample"));
  m_TimeSlider->setToolTip(tr("Set current time sample"));
  m_TimeSlider->setTracking(false);
  m_TimeSlider->setRange(0, 0);
  m_TimeSlider->setSingleStep(1);
  m_TimeSlider->setTickPosition(QSlider::TickPosition::TicksBelow);
  fullLayout->addWidget(m_TimeSlider);

  QObject::connect(m_TimeSlider, &QIntSlider::valueChanged, [this](int t) { this->OnTimeChanged(t); });

  scrollArea->setLayout(fullLayout);

  m_MainLayout.addWidget(scrollArea, 1, 0);
}

void
QTimelineWidget::onNewImage(Scene* s, std::string filepath)
{
  m_scene = s;
  m_filepath = filepath;

  int32_t minT = m_scene ? m_scene->m_timeLine.minTime() : 0;
  int32_t maxT = m_scene ? m_scene->m_timeLine.maxTime() : 0;

  m_TimeSlider->setRange(minT, maxT);
  m_TimeSlider->setValue(m_scene->m_timeLine.currentTime(), true);
  m_TimeSlider->setTickInterval((maxT - minT) / 10);
  m_TimeSlider->setSingleStep(1);

  // disable the slider if there is less than 2 time samples.
  m_TimeSlider->setEnabled(maxT > minT);
  this->parentWidget()->setWindowTitle(maxT > minT ? tr("Time") : tr("Time (disabled)"));
}

void
QTimelineWidget::OnTimeChanged(int newTime)
{
  if (!m_scene) {
    return;
  }
  if (m_scene->m_timeLine.currentTime() != newTime) {
    m_scene->m_timeLine.setCurrentTime(newTime);
    // assume a new time sample will have same exact channel configuration and dimensions as previous time.
    // we are just updating volume data.
    QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
    std::shared_ptr<ImageXYZC> image = FileReader::loadFromFile(m_filepath, nullptr, newTime, 0);
    QApplication::restoreOverrideCursor();
    if (!image) {
      // TODO FIXME if we fail to set the new time, then reset the GUI to previous time
      LOG_DEBUG << "Failed to open " << m_filepath;
      return;
    }
    m_scene->m_volume = image;

    // TODO update the AppearanceSettings channel gui with new Histograms
    for (uint32_t i = 0; i < m_scene->m_volume->sizeC(); ++i) {
      m_scene->m_volume->channel((uint32_t)i)->generateFromGradientData(m_scene->m_material.m_gradientData[i]);
    }

    m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(VolumeDataDirty);
    m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
  }
}

QTimelineDockWidget::QTimelineDockWidget(QWidget* parent, QRenderSettings* qrs)
  : QDockWidget(parent)
  , m_TimelineWidget(this, qrs)
{
  setWindowTitle(tr("Time"));

  m_TimelineWidget.setParent(this);

  setWidget(&m_TimelineWidget);
}
