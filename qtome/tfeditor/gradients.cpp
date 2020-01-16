/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the demonstration applications of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "gradients.h"
#include "hoverpoints.h"

#include "Controls.h"
#include "Defines.h"
#include "Logging.h"

#include <algorithm>

std::vector<std::pair<float, float>>
gradientStopsToVector(QGradientStops& stops)
{
  std::vector<std::pair<float, float>> v;
  for (int i = 0; i < stops.size(); ++i) {
    v.push_back(std::pair<float, float>(stops.at(i).first, stops.at(i).second.alphaF()));
  }
  return v;
}

QGradientStops
vectorToGradientStops(std::vector<std::pair<float, float>>& v)
{
  QGradientStops stops;
  for (int i = 0; i < v.size(); ++i) {
    stops.push_back(
      QPair<qreal, QColor>(v[i].first, QColor::fromRgbF(v[i].second, v[i].second, v[i].second, v[i].second)));
  }
  return stops;
}

ShadeWidget::ShadeWidget(const Histogram& histogram, ShadeType type, QWidget* parent)
  : QWidget(parent)
  , m_shade_type(type)
  , m_alpha_gradient(QLinearGradient(0, 0, 0, 0))
  , m_histogram(histogram)
{
  // Checkers background
  if (m_shade_type == ARGBShade) {
    QPixmap pm(20, 20);
    QPainter pmp(&pm);
    pmp.fillRect(0, 0, 10, 10, Qt::lightGray);
    pmp.fillRect(10, 10, 10, 10, Qt::lightGray);
    pmp.fillRect(0, 10, 10, 10, Qt::darkGray);
    pmp.fillRect(10, 0, 10, 10, Qt::darkGray);
    pmp.end();
    QPalette pal = palette();
    pal.setBrush(backgroundRole(), QBrush(pm));
    setAutoFillBackground(true);
    setPalette(pal);

  } else {
    setAttribute(Qt::WA_NoBackground);
  }

  QPolygonF points;
  points << QPointF(0, sizeHint().height()) << QPointF(sizeHint().width(), 0);

  m_hoverPoints = new HoverPoints(this, HoverPoints::CircleShape);
  //     m_hoverPoints->setConnectionType(HoverPoints::LineConnection);
  m_hoverPoints->setPoints(points);
  m_hoverPoints->setPointLock(0, HoverPoints::LockToLeft);
  m_hoverPoints->setPointLock(1, HoverPoints::LockToRight);
  m_hoverPoints->setSortType(HoverPoints::XSort);

  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

  connect(m_hoverPoints, &HoverPoints::pointsChanged, this, &ShadeWidget::colorsChanged);
}

QPolygonF
ShadeWidget::points() const
{
  return m_hoverPoints->points();
}

uint
ShadeWidget::colorAt(int x)
{
  generateShade();
  if (m_shade.isNull()) {
    return 0;
  }

  QPolygonF pts = m_hoverPoints->points();
  for (int i = 1; i < pts.size(); ++i) {
    if (pts.at(i - 1).x() <= x && pts.at(i).x() >= x) {
      QLineF l(pts.at(i - 1), pts.at(i));
      l.setLength(l.length() * ((x - l.x1()) / l.dx()));
      return m_shade.pixel(qRound(qMin(l.x2(), (qreal(m_shade.width() - 1)))),
                           qRound(qMin(l.y2(), qreal(m_shade.height() - 1))));
    }
  }
  return 0;
}

void
ShadeWidget::setGradientStops(const QGradientStops& stops)
{
  if (m_shade_type == ARGBShade) {
    m_alpha_gradient = QLinearGradient(0, 0, width(), 0);

    for (int i = 0; i < stops.size(); ++i) {
      QColor c = stops.at(i).second;
      m_alpha_gradient.setColorAt(stops.at(i).first, QColor(c.red(), c.green(), c.blue()));
    }

    m_shade = QImage();
    generateShade();
    update();
  }
}

void
ShadeWidget::paintEvent(QPaintEvent*)
{
  generateShade();

  QPainter p(this);
  p.drawImage(0, 0, m_shade);
  /*
        qreal barWidth = width() / (qreal)m_histogram.size();

      for (int i = 0; i < m_histogram.size(); ++i) {
          qreal h = m_histogram[i] * height();
          // draw level
          painter.fillRect(barWidth * i, height() - h, barWidth * (i + 1), height(), Qt::red);
          // clear the rest of the control
          painter.fillRect(barWidth * i, 0, barWidth * (i + 1), height() - h, Qt::black);
      }
  */
  p.setPen(QColor(146, 146, 146));
  p.drawRect(0, 0, width() - 1, height() - 1);
}

void
ShadeWidget::drawHistogram(QPainter& p, int w, int h)
{
  size_t nbins = m_histogram._bins.size();
  int maxbinsize = m_histogram._bins[m_histogram._maxBin];
  for (size_t i = 0; i < nbins; ++i) {
    float binheight = (float)m_histogram._bins[i] * (float)(h - 1) / (float)maxbinsize;
    p.fillRect(
      QRectF((float)i * (float)(w - 1) / (float)nbins, h - 1 - binheight, (float)(w - 1) / (float)nbins, binheight),
      QColor(0, 0, 0, 255));
  }
}

void
ShadeWidget::generateShade()
{
  if (m_shade.isNull() || m_shade.size() != size()) {

    QRect qrect = rect();
    QSize qsize = size();
    if (m_shade_type == ARGBShade) {
      m_shade = QImage(qsize, QImage::Format_ARGB32_Premultiplied);
      if (m_shade.isNull()) {
        return;
      }
      m_shade.fill(0);

      QPainter p(&m_shade);
      p.fillRect(qrect, m_alpha_gradient);

      p.setCompositionMode(QPainter::CompositionMode_DestinationIn);
      QLinearGradient fade(0, 0, 0, height() - 1);
      fade.setColorAt(0, QColor(255, 255, 255, 255));
      fade.setColorAt(1, QColor(0, 0, 0, 0));
      p.fillRect(qrect, fade);

      p.setCompositionMode(QPainter::CompositionMode_SourceOver);

      drawHistogram(p, qsize.width(), qsize.height());

    } else {
      m_shade = QImage(qsize, QImage::Format_RGB32);
      if (m_shade.isNull()) {
        return;
      }
      QLinearGradient shade(0, 0, 0, height());
      shade.setColorAt(1, Qt::black);

      if (m_shade_type == RedShade)
        shade.setColorAt(0, Qt::red);
      else if (m_shade_type == GreenShade)
        shade.setColorAt(0, Qt::green);
      else
        shade.setColorAt(0, Qt::blue);

      QPainter p(&m_shade);
      p.fillRect(qrect, shade);

      p.setCompositionMode(QPainter::CompositionMode_SourceOver);

      drawHistogram(p, qsize.width(), qsize.height());
    }
  }
}

GradientEditor::GradientEditor(const Histogram& histogram, QWidget* parent)
  : QWidget(parent)
{
  QVBoxLayout* vbox = new QVBoxLayout(this);
  vbox->setSpacing(1);
  vbox->setMargin(1);

  m_alpha_shade = new ShadeWidget(histogram, ShadeWidget::ARGBShade, this);

  vbox->addWidget(m_alpha_shade);

  connect(m_alpha_shade, &ShadeWidget::colorsChanged, this, &GradientEditor::pointsUpdated);
}

inline static bool
x_less_than(const QPointF& p1, const QPointF& p2)
{
  return p1.x() < p2.x();
}

void
GradientEditor::pointsUpdated()
{
  qreal w = m_alpha_shade->width();

  QGradientStops stops;

  QPolygonF points;

  points += m_alpha_shade->points();

  std::sort(points.begin(), points.end(), x_less_than);

  for (int i = 0; i < points.size(); ++i) {
    qreal x = int(points.at(i).x());
    if (i + 1 < points.size() && x == points.at(i + 1).x())
      continue;
    unsigned int pixelvalue = m_alpha_shade->colorAt(int(x));
    // TODO let each point in m_alpha_shade have a full RGBA color and use a color picker to assign it via dbl click or
    // some other means
    unsigned int r = (0x00ff0000 & pixelvalue) >> 16;
    unsigned int g = (0x0000ff00 & pixelvalue) >> 8;
    unsigned int b = (0x000000ff & pixelvalue);
    unsigned int a = (0xff000000 & pixelvalue) >> 24;
    QColor color(r, g, b, a);

    if (x / w > 1)
      return;

    stops << QGradientStop(x / w, color);
  }

  m_alpha_shade->setGradientStops(stops);

  emit gradientStopsChanged(stops);
}

static void
set_shade_points(const QPolygonF& points, ShadeWidget* shade)
{
  shade->hoverPoints()->setPoints(points);
  shade->hoverPoints()->setPointLock(0, HoverPoints::LockToLeft);
  shade->hoverPoints()->setPointLock(points.size() - 1, HoverPoints::LockToRight);
  shade->update();
}

void
GradientEditor::setGradientStops(const QGradientStops& stops)
{
  QPolygonF pts_red, pts_green, pts_blue, pts_alpha;

  qreal h_alpha = m_alpha_shade->height();

  for (int i = 0; i < stops.size(); ++i) {
    qreal pos = stops.at(i).first;
    QRgb color = stops.at(i).second.rgba();
    pts_alpha << QPointF(pos * m_alpha_shade->width(), h_alpha - qAlpha(color) * h_alpha / 255);
  }

  set_shade_points(pts_alpha, m_alpha_shade);
}

GradientWidget::GradientWidget(const Histogram& histogram, QWidget* parent)
  : QWidget(parent)
  , m_histogram(histogram)
{
  setWindowTitle(tr("Gradients"));

  // QGroupBox* editorGroup = new QGroupBox(this);
  // editorGroup->setTitle(tr("Color Editor"));
  m_editor = new GradientEditor(m_histogram, this);

  auto* sectionLayout = Controls::createFormLayout();

  QButtonGroup* btnGroup = new QButtonGroup(this);
  QPushButton* windowLevelButton = new QPushButton("Wnd/Lvl");
  windowLevelButton->setToolTip("Window/Level");
  windowLevelButton->setStatusTip("Choose Window/Level mode");
  QPushButton* isoButton = new QPushButton("Iso");
  isoButton->setToolTip("Isovalue");
  isoButton->setStatusTip("Choose Isovalue mode");
  QPushButton* pctButton = new QPushButton("Pct");
  pctButton->setToolTip("Histogram Percentiles");
  pctButton->setStatusTip("Choose Histogram percentiles mode");
  QPushButton* customButton = new QPushButton("Custom");
  customButton->setToolTip("Custom");
  customButton->setStatusTip("Choose Custom editing mode");

  btnGroup->addButton(windowLevelButton, 1);
  btnGroup->addButton(isoButton, 2);
  btnGroup->addButton(pctButton, 3);
  btnGroup->addButton(customButton, 4);
  QHBoxLayout* hbox = new QHBoxLayout();
  hbox->setSpacing(0);
  for (auto btn : btnGroup->buttons()) {
    btn->setCheckable(true);
    hbox->addWidget(btn);
  }

  QWidget* firstPageWidget = new QWidget;
  auto* section0Layout = Controls::createFormLayout();
  firstPageWidget->setLayout(section0Layout);

  QWidget* secondPageWidget = new QWidget;
  auto* section1Layout = Controls::createFormLayout();
  secondPageWidget->setLayout(section1Layout);

  QWidget* thirdPageWidget = new QWidget;
  auto* section2Layout = Controls::createFormLayout();
  thirdPageWidget->setLayout(section2Layout);

  QWidget* fourthPageWidget = new QWidget;
  auto* section3Layout = Controls::createFormLayout();
  fourthPageWidget->setLayout(section3Layout);

  QStackedLayout* stackedLayout = new QStackedLayout;
  stackedLayout->addWidget(firstPageWidget);
  stackedLayout->addWidget(secondPageWidget);
  stackedLayout->addWidget(thirdPageWidget);
  stackedLayout->addWidget(fourthPageWidget);

  connect(btnGroup, QOverload<int>::of(&QButtonGroup::buttonClicked), [this, stackedLayout](int id) {
    GradientEditMode modeToSet;
    switch (id) {
      case 1:
        modeToSet = GradientEditMode::WINDOW_LEVEL;
        stackedLayout->setCurrentIndex(0);
        this->onSetWindowLevel(this->m_gradientData.m_window, this->m_gradientData.m_level);
        break;
      case 2:
        modeToSet = GradientEditMode::ISOVALUE;
        stackedLayout->setCurrentIndex(1);
        this->onSetIsovalue(this->m_gradientData.m_isovalue, this->m_gradientData.m_isorange);
        break;
      case 3:
        modeToSet = GradientEditMode::PERCENTILE;
        stackedLayout->setCurrentIndex(2);
        this->onSetHistogramPercentiles(this->m_gradientData.m_pctLow, this->m_gradientData.m_pctHigh);
        break;
      case 4: {
        modeToSet = GradientEditMode::CUSTOM;
        stackedLayout->setCurrentIndex(3);

        QGradientStops stops = vectorToGradientStops(this->m_gradientData.m_customControlPoints);
        m_editor->setGradientStops(stops);
        emit gradientStopsChanged(stops);

        // this->onSetControlPoints(this->m_gradientData.m_customControlPoints);

      } break;
      default:
        LOG_ERROR << "Bad button id for gradient editor mode";
        break;
    }
    // if not current mode, then set mode and update:
    if (this->m_gradientData.m_editMode != modeToSet) {
      this->m_gradientData.m_editMode = modeToSet;
      // assumes button ids are same values as stacked widget indices
      stackedLayout->setCurrentIndex(id - 1);
      // update graph
      this->m_gradientData.m_editMode = modeToSet;
    }
  });

  QNumericSlider* windowSlider = new QNumericSlider();
  windowSlider->setStatusTip("Window");
  windowSlider->setToolTip("Window");
  windowSlider->setRange(0.0, 1.0);
  windowSlider->setValue(0.25);
  section0Layout->addRow("Window", windowSlider);
  QNumericSlider* levelSlider = new QNumericSlider();
  levelSlider->setStatusTip("Level");
  levelSlider->setToolTip("Level");
  levelSlider->setRange(0.0, 1.0);
  levelSlider->setValue(0.5);
  section0Layout->addRow("Level", levelSlider);
  connect(windowSlider, &QNumericSlider::valueChanged, [this, levelSlider](double d) {
    this->m_gradientData.m_window = d;
    this->onSetWindowLevel(d, levelSlider->value());
  });
  connect(levelSlider, &QNumericSlider::valueChanged, [this, windowSlider](double d) {
    this->m_gradientData.m_level = d;
    this->onSetWindowLevel(windowSlider->value(), d);
  });

  QNumericSlider* isovalueSlider = new QNumericSlider();
  isovalueSlider->setStatusTip("Isovalue");
  isovalueSlider->setToolTip("Set Isovalue");
  isovalueSlider->setRange(0.0, 1.0);
  isovalueSlider->setValue(0.5);
  section1Layout->addRow("Isovalue", isovalueSlider);
  QNumericSlider* isorangeSlider = new QNumericSlider();
  isorangeSlider->setStatusTip("Isovalue range");
  isorangeSlider->setToolTip("Set range above and below isovalue");
  isorangeSlider->setRange(0.0, 1.0);
  isorangeSlider->setValue(0.01);
  section1Layout->addRow("Iso-range", isorangeSlider);
  connect(isovalueSlider, &QNumericSlider::valueChanged, [this, isorangeSlider](double d) {
    this->m_gradientData.m_isovalue = d;
    this->onSetIsovalue(d, isorangeSlider->value());
  });
  connect(isorangeSlider, &QNumericSlider::valueChanged, [this, isovalueSlider](double d) {
    this->m_gradientData.m_isorange = d;
    this->onSetIsovalue(isovalueSlider->value(), d);
  });

  QNumericSlider* pctLowSlider = new QNumericSlider();
  pctLowSlider->setStatusTip("Low percentile");
  pctLowSlider->setToolTip("Set bottom percentile");
  pctLowSlider->setRange(0.0, 1.0);
  pctLowSlider->setValue(0.5);
  section2Layout->addRow("Pct Min", pctLowSlider);
  QNumericSlider* pctHighSlider = new QNumericSlider();
  pctHighSlider->setStatusTip("High percentile");
  pctHighSlider->setToolTip("Set top percentile");
  pctHighSlider->setRange(0.0, 1.0);
  pctHighSlider->setValue(0.98);
  section2Layout->addRow("Pct Max", pctHighSlider);
  connect(pctLowSlider, &QNumericSlider::valueChanged, [this, pctHighSlider](double d) {
    this->m_gradientData.m_pctLow = d;
    this->onSetHistogramPercentiles(d, pctHighSlider->value());
  });
  connect(pctHighSlider, &QNumericSlider::valueChanged, [this, pctLowSlider](double d) {
    this->m_gradientData.m_pctHigh = d;
    this->onSetHistogramPercentiles(pctLowSlider->value(), d);
  });

  // Layouts

  QVBoxLayout* mainGroupLayout = new QVBoxLayout(this);
  mainGroupLayout->addWidget(m_editor);
  mainGroupLayout->addLayout(hbox);
  mainGroupLayout->addLayout(stackedLayout);
  mainGroupLayout->addLayout(sectionLayout);
  mainGroupLayout->addStretch(1);

  connect(m_editor, &GradientEditor::gradientStopsChanged, this, &GradientWidget::onGradientStopsChanged);

  QTimer::singleShot(50, this, SLOT(setDefault4()));
}

void
GradientWidget::onGradientStopsChanged(const QGradientStops& stops)
{
  // update the data stored in m_gradientData
  m_gradientData.m_customControlPoints.clear();
  for (int i = 0; i < stops.size(); ++i) {
    m_gradientData.m_customControlPoints.push_back(
      std::pair<float, float>(stops.at(i).first, stops.at(i).second.alphaF()));
  }

  emit gradientStopsChanged(stops);
}

void
GradientWidget::onSetHistogramPercentiles(float pctLow, float pctHigh)
{
  float window, level;
  m_histogram.computeWindowLevelFromPercentiles(pctLow, pctHigh, window, level);
  this->onSetWindowLevel(window, level);
}

void
GradientWidget::onSetWindowLevel(float window, float level)
{
  QGradientStops stops;
  QPolygonF points;
  float lowEnd = level - window * 0.5;
  float highEnd = level + window * 0.5;
  if (lowEnd <= 0.0) {
    float val = -lowEnd / (highEnd - lowEnd);
    stops << QGradientStop(0.0, QColor::fromRgbF(val, val, val, val));
  } else {
    stops << QGradientStop(0.0, QColor::fromRgba(0));
    stops << QGradientStop(lowEnd, QColor::fromRgba(0));
  }
  if (highEnd >= 1.0) {
    float val = (1.0 - lowEnd) / (highEnd - lowEnd);
    stops << QGradientStop(1.0, QColor::fromRgbF(val, val, val, val));
  } else {
    stops << QGradientStop(highEnd, QColor::fromRgba(0xffffffff));
    stops << QGradientStop(1.0, QColor::fromRgba(0xffffffff));
  }
  m_editor->setGradientStops(stops);
  emit gradientStopsChanged(stops);
}

void
GradientWidget::onSetIsovalue(float isovalue, float width)
{
  QGradientStops stops;
  QPolygonF points;
  float lowEnd = isovalue - width * 0.5;
  float highEnd = isovalue + width * 0.5;
  stops << QGradientStop(0.00, QColor::fromRgba(0));
  stops << QGradientStop(lowEnd, QColor::fromRgba(0));
  stops << QGradientStop(lowEnd, QColor::fromRgba(0xffffffff));
  stops << QGradientStop(highEnd, QColor::fromRgba(0xffffffff));
  stops << QGradientStop(highEnd, QColor::fromRgba(0));
  stops << QGradientStop(1.0, QColor::fromRgba(0));
  m_editor->setGradientStops(stops);
  m_gradientData.m_isovalue = isovalue;
  m_gradientData.m_isorange = width;
  emit gradientStopsChanged(stops);
}

void
GradientWidget::setDefault(int config)
{
  QGradientStops stops;
  QPolygonF points;
  switch (config) {
    case 1:
      stops << QGradientStop(0.00, QColor::fromRgba(0));
      stops << QGradientStop(0.04, QColor::fromRgba(0xff131360));
      stops << QGradientStop(0.08, QColor::fromRgba(0xff202ccc));
      stops << QGradientStop(0.42, QColor::fromRgba(0xff93d3f9));
      stops << QGradientStop(0.51, QColor::fromRgba(0xffb3e6ff));
      stops << QGradientStop(0.73, QColor::fromRgba(0xffffffec));
      stops << QGradientStop(0.92, QColor::fromRgba(0xff5353d9));
      stops << QGradientStop(0.96, QColor::fromRgba(0xff262666));
      stops << QGradientStop(1.00, QColor::fromRgba(0));
      break;

    case 2:
      stops << QGradientStop(0.00, QColor::fromRgba(0xffffffff));
      stops << QGradientStop(0.11, QColor::fromRgba(0xfff9ffa0));
      stops << QGradientStop(0.13, QColor::fromRgba(0xfff9ff99));
      stops << QGradientStop(0.14, QColor::fromRgba(0xfff3ff86));
      stops << QGradientStop(0.49, QColor::fromRgba(0xff93b353));
      stops << QGradientStop(0.87, QColor::fromRgba(0xff264619));
      stops << QGradientStop(0.96, QColor::fromRgba(0xff0c1306));
      stops << QGradientStop(1.00, QColor::fromRgba(0));
      break;

    case 3:
      stops << QGradientStop(0.00, QColor::fromRgba(0));
      stops << QGradientStop(0.10, QColor::fromRgba(0xffe0cc73));
      stops << QGradientStop(0.17, QColor::fromRgba(0xffc6a006));
      stops << QGradientStop(0.46, QColor::fromRgba(0xff600659));
      stops << QGradientStop(0.72, QColor::fromRgba(0xff0680ac));
      stops << QGradientStop(0.92, QColor::fromRgba(0xffb9d9e6));
      stops << QGradientStop(1.00, QColor::fromRgba(0));
      break;

    case 4:
      stops << QGradientStop(0.00, QColor::fromRgba(0xff000000));
      stops << QGradientStop(1.00, QColor::fromRgba(0xffffffff));
      break;

    default:
      qWarning("bad default: %d\n", config);
      break;
  }

  m_editor->setGradientStops(stops);
}

void
GradientWidget::updatePresetName()
{
  QMetaEnum presetEnum = QMetaEnum::fromType<QGradient::Preset>();
  m_presetButton->setText(QLatin1String(presetEnum.key(m_presetIndex)));
}

void
GradientWidget::changePresetBy(int indexOffset)
{
  QMetaEnum presetEnum = QMetaEnum::fromType<QGradient::Preset>();
  m_presetIndex = qBound(0, m_presetIndex + indexOffset, presetEnum.keyCount() - 1);

  QGradient::Preset preset = static_cast<QGradient::Preset>(presetEnum.value(m_presetIndex));
  QGradient gradient(preset);
  if (gradient.type() != QGradient::LinearGradient)
    return;

  m_editor->setGradientStops(gradient.stops());

  updatePresetName();
}
