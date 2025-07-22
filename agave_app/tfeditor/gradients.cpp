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
#include "renderlib/Defines.h"
#include "renderlib/Logging.h"
#include "renderlib/MathUtil.h"

#include <algorithm>

std::vector<LutControlPoint>
gradientStopsToVector(QGradientStops& stops)
{
  std::vector<LutControlPoint> v;
  for (int i = 0; i < stops.size(); ++i) {
    v.push_back(LutControlPoint(stops.at(i).first, stops.at(i).second.alphaF()));
  }
  return v;
}

QGradientStops
vectorToGradientStops(std::vector<LutControlPoint>& v)
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
    setAttribute(Qt::WA_OpaquePaintEvent);
  }

  QPolygonF points;
  points << QPointF(0.0f, 0.0f) << QPointF(1.0f, 1.0f);

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

void
ShadeWidget::setEditable(bool editable)
{
  m_hoverPoints->setEditable(editable);
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
  // vbox->setMargin(1);

  m_alpha_shade = new ShadeWidget(histogram, ShadeWidget::ARGBShade, this);

  vbox->addWidget(m_alpha_shade);

  connect(m_alpha_shade, &ShadeWidget::colorsChanged, this, &GradientEditor::pointsUpdated);
}

inline static bool
x_less_than(const QPointF& p1, const QPointF& p2)
{
  return p1.x() < p2.x();
}

inline static bool
controlpoint_x_less_than(const LutControlPoint& p1, const LutControlPoint& p2)
{
  return p1.first < p2.first;
}

QGradientStops
pointsToGradientStops(QPolygonF points)
{
  QGradientStops stops;
  std::sort(points.begin(), points.end(), x_less_than);

  for (int i = 0; i < points.size(); ++i) {
    qreal x = points.at(i).x();
    if (i + 1 < points.size() && x == points.at(i + 1).x())
      continue;
    float pixelvalue = points.at(i).y();
    // TODO future: let each point in m_alpha_shade have a full RGBA color and use a color picker to assign it via dbl
    // click or some other means
    // unsigned int pixelvalue = m_alpha_shade->colorAt(int(x));
    // unsigned int r = (0x00ff0000 & pixelvalue) >> 16;
    // unsigned int g = (0x0000ff00 & pixelvalue) >> 8;
    // unsigned int b = (0x000000ff & pixelvalue);
    // unsigned int a = (0xff000000 & pixelvalue) >> 24;
    // QColor color(r, g, b, a);

    QColor color = QColor::fromRgbF(pixelvalue, pixelvalue, pixelvalue, pixelvalue);
    if (x > 1) {
      LOG_ERROR << "control point x greater than 1";
      return stops;
    }

    stops << QGradientStop(x, color);
  }
  return stops;
}

void
GradientEditor::pointsUpdated()
{
  // qreal w = m_alpha_shade->width();

  QGradientStops stops = pointsToGradientStops(m_alpha_shade->points());

  m_alpha_shade->setGradientStops(stops);

  emit gradientStopsChanged(stops);
}

static void
set_shade_points(const QPolygonF& points, ShadeWidget* shade)
{
  if (points.size() < 2) {
    return;
  }

  QGradientStops stops = pointsToGradientStops(points);
  shade->setGradientStops(stops);

  shade->hoverPoints()->setPoints(points);
  shade->hoverPoints()->setPointLock(0, HoverPoints::LockToLeft);
  shade->hoverPoints()->setPointLock(points.size() - 1, HoverPoints::LockToRight);
  shade->update();
}

void
GradientEditor::setControlPoints(const std::vector<LutControlPoint>& points)
{
  QPolygonF pts_alpha;

  for (auto p : points) {
    pts_alpha << QPointF(p.first, p.second);
  }

  set_shade_points(pts_alpha, m_alpha_shade);
}

void
GradientEditor::wheelEvent(QWheelEvent* event)
{
  // wheel does nothing here!
  event->ignore();
}

GradientWidget::GradientWidget(const Histogram& histogram, GradientData* dataObject, QWidget* parent)
  : QWidget(parent)
  , m_histogram(histogram)
  , m_gradientData(dataObject)
{
  QVBoxLayout* mainGroupLayout = new QVBoxLayout(this);

  // setWindowTitle(tr("Gradients"));

  // QGroupBox* editorGroup = new QGroupBox(this);
  // editorGroup->setTitle(tr("Color Editor"));
  m_editor = new GradientEditor(m_histogram, this);
  mainGroupLayout->addWidget(m_editor);

  auto* sectionLayout = Controls::createAgaveFormLayout();

  QButtonGroup* btnGroup = new QButtonGroup(this);
  QPushButton* minMaxButton = new QPushButton("Min/Max");
  minMaxButton->setToolTip(tr("Min/Max"));
  minMaxButton->setStatusTip(tr("Choose Min/Max mode"));
  QPushButton* windowLevelButton = new QPushButton("Wnd/Lvl");
  windowLevelButton->setToolTip(tr("Window/Level"));
  windowLevelButton->setStatusTip(tr("Choose Window/Level mode"));
  QPushButton* isoButton = new QPushButton("Iso");
  isoButton->setToolTip(tr("Isovalue"));
  isoButton->setStatusTip(tr("Choose Isovalue mode"));
  QPushButton* pctButton = new QPushButton("Pct");
  pctButton->setToolTip(tr("Histogram Percentiles"));
  pctButton->setStatusTip(tr("Choose Histogram percentiles mode"));
  QPushButton* customButton = new QPushButton("Custom");
  customButton->setToolTip(tr("Custom"));
  customButton->setStatusTip(tr("Choose Custom editing mode"));

  static const int WINDOW_LEVEL_BTNID = 1;
  static const int ISO_BTNID = 2;
  static const int PCT_BTNID = 3;
  static const int CUSTOM_BTNID = 4;
  static const int MINMAX_BTNID = 5;
  static std::map<int, GradientEditMode> btnIdToGradientMode = { { WINDOW_LEVEL_BTNID, GradientEditMode::WINDOW_LEVEL },
                                                                 { ISO_BTNID, GradientEditMode::ISOVALUE },
                                                                 { PCT_BTNID, GradientEditMode::PERCENTILE },
                                                                 { MINMAX_BTNID, GradientEditMode::MINMAX },
                                                                 { CUSTOM_BTNID, GradientEditMode::CUSTOM } };
  static std::map<GradientEditMode, int> gradientModeToBtnId = { { GradientEditMode::WINDOW_LEVEL, WINDOW_LEVEL_BTNID },
                                                                 { GradientEditMode::ISOVALUE, ISO_BTNID },
                                                                 { GradientEditMode::PERCENTILE, PCT_BTNID },
                                                                 { GradientEditMode::MINMAX, MINMAX_BTNID },
                                                                 { GradientEditMode::CUSTOM, CUSTOM_BTNID } };
  static std::map<int, int> btnIdToStackedPage = {
    { WINDOW_LEVEL_BTNID, 1 }, { ISO_BTNID, 2 }, { PCT_BTNID, 3 }, { MINMAX_BTNID, 0 }, { CUSTOM_BTNID, 4 }
  };
  btnGroup->addButton(minMaxButton, MINMAX_BTNID);
  btnGroup->addButton(windowLevelButton, WINDOW_LEVEL_BTNID);
  btnGroup->addButton(isoButton, ISO_BTNID);
  btnGroup->addButton(pctButton, PCT_BTNID);
  btnGroup->addButton(customButton, CUSTOM_BTNID);
  QHBoxLayout* hbox = new QHBoxLayout();
  hbox->setSpacing(0);

  int initialButtonId = WINDOW_LEVEL_BTNID;
  GradientEditMode m = m_gradientData->m_activeMode;
  initialButtonId = gradientModeToBtnId[m];

  for (auto btn : btnGroup->buttons()) {
    btn->setCheckable(true);
    // set checked state initially.
    int btnid = btnGroup->id(btn);
    if (btnid == initialButtonId) {
      btn->setChecked(true);
    }
    hbox->addWidget(btn);
  }
  mainGroupLayout->addLayout(hbox);

  QWidget* firstPageWidget = new QWidget;
  auto* section0Layout = Controls::createAgaveFormLayout();
  firstPageWidget->setLayout(section0Layout);

  QWidget* secondPageWidget = new QWidget;
  auto* section1Layout = Controls::createAgaveFormLayout();
  secondPageWidget->setLayout(section1Layout);

  QWidget* thirdPageWidget = new QWidget;
  auto* section2Layout = Controls::createAgaveFormLayout();
  thirdPageWidget->setLayout(section2Layout);

  QWidget* fourthPageWidget = new QWidget;
  auto* section3Layout = Controls::createAgaveFormLayout();
  fourthPageWidget->setLayout(section3Layout);

  QWidget* fifthPageWidget = new QWidget;
  auto* section4Layout = Controls::createAgaveFormLayout();
  fifthPageWidget->setLayout(section4Layout);

  QStackedLayout* stackedLayout = new QStackedLayout(mainGroupLayout);
  stackedLayout->addWidget(firstPageWidget);
  stackedLayout->addWidget(secondPageWidget);
  stackedLayout->addWidget(thirdPageWidget);
  stackedLayout->addWidget(fourthPageWidget);
  stackedLayout->addWidget(fifthPageWidget);

  int initialStackedPageIndex = btnIdToStackedPage[initialButtonId];
  stackedLayout->setCurrentIndex(initialStackedPageIndex);
  // if this is not custom mode, then disable the gradient editor
  m_editor->setEditable(m == GradientEditMode::CUSTOM);

  connect(btnGroup,
          QOverload<QAbstractButton*>::of(&QButtonGroup::buttonClicked),
          [this, btnGroup, stackedLayout](QAbstractButton* button) {
            int id = btnGroup->id(button);
            GradientEditMode modeToSet = btnIdToGradientMode[id];
            // if mode is not changing, we are done.
            if (modeToSet == this->m_gradientData->m_activeMode) {
              return;
            }
            this->m_gradientData->m_activeMode = modeToSet;

            stackedLayout->setCurrentIndex(btnIdToStackedPage[id]);

            // if this is not custom mode, then disable the gradient editor
            m_editor->setEditable(modeToSet == GradientEditMode::CUSTOM);

            this->forceDataUpdate();
          });

  QIntSlider* minu16Slider = new QIntSlider();
  minu16Slider->setStatusTip(tr("Minimum u16 value"));
  minu16Slider->setToolTip(tr("Set minimum u16 value"));
  minu16Slider->setRange(0, 65535);
  minu16Slider->setSingleStep(1);
  minu16Slider->setValue(m_gradientData->m_minu16);
  section0Layout->addRow("Min u16", minu16Slider);
  QIntSlider* maxu16Slider = new QIntSlider();
  maxu16Slider->setStatusTip(tr("Maximum u16 value"));
  maxu16Slider->setToolTip(tr("Set maximum u16 value"));
  maxu16Slider->setRange(0, 65535);
  maxu16Slider->setSingleStep(1);
  maxu16Slider->setValue(m_gradientData->m_maxu16);
  section0Layout->addRow("Max u16", maxu16Slider);
  connect(minu16Slider, &QIntSlider::valueChanged, [this, maxu16Slider](int i) {
    this->m_gradientData->m_minu16 = i;
    this->onSetMinMax(i, this->m_gradientData->m_maxu16);
  });
  connect(maxu16Slider, &QIntSlider::valueChanged, [this, minu16Slider](int i) {
    this->m_gradientData->m_maxu16 = i;
    this->onSetMinMax(this->m_gradientData->m_minu16, i);
  });

  QNumericSlider* windowSlider = new QNumericSlider();
  windowSlider->setStatusTip(tr("Window"));
  windowSlider->setToolTip(tr("Set size of range of intensities"));
  windowSlider->setRange(0.0, 1.0);
  windowSlider->setSingleStep(0.01);
  windowSlider->setDecimals(3);
  windowSlider->setValue(m_gradientData->m_window);
  section1Layout->addRow("Window", windowSlider);
  QNumericSlider* levelSlider = new QNumericSlider();
  levelSlider->setStatusTip(tr("Level"));
  levelSlider->setToolTip(tr("Set level of mid intensity"));
  levelSlider->setRange(0.0, 1.0);
  levelSlider->setSingleStep(0.01);
  levelSlider->setDecimals(3);
  levelSlider->setValue(m_gradientData->m_level);
  section1Layout->addRow("Level", levelSlider);
  connect(windowSlider, &QNumericSlider::valueChanged, [this, levelSlider](double d) {
    this->m_gradientData->m_window = d;
    this->onSetWindowLevel(d, levelSlider->value());
  });
  connect(levelSlider, &QNumericSlider::valueChanged, [this, windowSlider](double d) {
    this->m_gradientData->m_level = d;
    this->onSetWindowLevel(windowSlider->value(), d);
  });

  QNumericSlider* isovalueSlider = new QNumericSlider();
  isovalueSlider->setStatusTip(tr("Isovalue"));
  isovalueSlider->setToolTip(tr("Set Isovalue"));
  isovalueSlider->setRange(0.0, 1.0);
  isovalueSlider->setSingleStep(0.01);
  isovalueSlider->setDecimals(3);
  isovalueSlider->setValue(m_gradientData->m_isovalue);
  section2Layout->addRow("Isovalue", isovalueSlider);
  QNumericSlider* isorangeSlider = new QNumericSlider();
  isorangeSlider->setStatusTip(tr("Isovalue range"));
  isorangeSlider->setToolTip(tr("Set range above and below isovalue"));
  isorangeSlider->setRange(0.0, 1.0);
  isorangeSlider->setSingleStep(0.01);
  isorangeSlider->setDecimals(3);
  isorangeSlider->setValue(m_gradientData->m_isorange);
  section2Layout->addRow("Iso-range", isorangeSlider);
  connect(isovalueSlider, &QNumericSlider::valueChanged, [this, isorangeSlider](double d) {
    this->m_gradientData->m_isovalue = d;
    this->onSetIsovalue(d, isorangeSlider->value());
  });
  connect(isorangeSlider, &QNumericSlider::valueChanged, [this, isovalueSlider](double d) {
    this->m_gradientData->m_isorange = d;
    this->onSetIsovalue(isovalueSlider->value(), d);
  });

  QNumericSlider* pctLowSlider = new QNumericSlider();
  pctLowSlider->setStatusTip(tr("Low percentile"));
  pctLowSlider->setToolTip(tr("Set bottom percentile"));
  pctLowSlider->setRange(0.0, 1.0);
  pctLowSlider->setSingleStep(0.01);
  pctLowSlider->setDecimals(3);
  pctLowSlider->setValue(m_gradientData->m_pctLow);
  section3Layout->addRow("Pct Min", pctLowSlider);
  QNumericSlider* pctHighSlider = new QNumericSlider();
  pctHighSlider->setStatusTip(tr("High percentile"));
  pctHighSlider->setToolTip(tr("Set top percentile"));
  pctHighSlider->setRange(0.0, 1.0);
  pctHighSlider->setSingleStep(0.01);
  pctHighSlider->setDecimals(3);
  pctHighSlider->setValue(m_gradientData->m_pctHigh);
  section3Layout->addRow("Pct Max", pctHighSlider);
  connect(pctLowSlider, &QNumericSlider::valueChanged, [this, pctHighSlider](double d) {
    this->m_gradientData->m_pctLow = d;
    this->onSetHistogramPercentiles(d, pctHighSlider->value());
  });
  connect(pctHighSlider, &QNumericSlider::valueChanged, [this, pctLowSlider](double d) {
    this->m_gradientData->m_pctHigh = d;
    this->onSetHistogramPercentiles(pctLowSlider->value(), d);
  });

  mainGroupLayout->addLayout(sectionLayout);
  mainGroupLayout->addStretch(1);

  connect(m_editor, &GradientEditor::gradientStopsChanged, this, &GradientWidget::onGradientStopsChanged);

  forceDataUpdate();
}

void
GradientWidget::forceDataUpdate()
{
  GradientEditMode mode = this->m_gradientData->m_activeMode;

  switch (mode) {
    case GradientEditMode::WINDOW_LEVEL:
      this->onSetWindowLevel(this->m_gradientData->m_window, this->m_gradientData->m_level);
      break;
    case GradientEditMode::ISOVALUE:
      this->onSetIsovalue(this->m_gradientData->m_isovalue, this->m_gradientData->m_isorange);
      break;
    case GradientEditMode::PERCENTILE:
      this->onSetHistogramPercentiles(this->m_gradientData->m_pctLow, this->m_gradientData->m_pctHigh);
      break;
    case GradientEditMode::MINMAX:
      this->onSetMinMax(this->m_gradientData->m_minu16, this->m_gradientData->m_maxu16);
      break;
    case GradientEditMode::CUSTOM: {
      m_editor->setControlPoints(this->m_gradientData->m_customControlPoints);
      QGradientStops stops = vectorToGradientStops(this->m_gradientData->m_customControlPoints);
      emit gradientStopsChanged(stops);
    } break;
    default:
      LOG_ERROR << "Bad gradient editor mode";
      break;
  }
}

void
GradientWidget::onGradientStopsChanged(const QGradientStops& stops)
{
  // update the data stored in m_gradientData
  m_gradientData->m_customControlPoints.clear();
  for (int i = 0; i < stops.size(); ++i) {
    m_gradientData->m_customControlPoints.push_back(LutControlPoint(stops.at(i).first, stops.at(i).second.alphaF()));
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
  std::vector<LutControlPoint> points;
  static const float epsilon = 0.000001f;
  window = std::max(window, epsilon);
  float lowEnd = level - window * 0.5f;
  float highEnd = level + window * 0.5f;
  if (lowEnd <= 0.0f) {
    float val = -lowEnd / (highEnd - lowEnd);
    points.push_back({ 0.0f, val });
  } else {
    points.push_back({ 0.0f, 0.0f });
    points.push_back({ lowEnd, 0.0f });
  }
  if (highEnd >= 1.0f) {
    float val = (1.0f - lowEnd) / (highEnd - lowEnd);
    points.push_back({ 1.0f, val });
  } else {
    points.push_back({ highEnd, 1.0f });
    points.push_back({ 1.0f, 1.0f });
  }
  m_editor->setControlPoints(points);
  emit gradientStopsChanged(vectorToGradientStops(points));
}

void
GradientWidget::onSetMinMax(uint16_t minu16, uint16_t maxu16)
{
  float relativeMin = normalizeInt(minu16);
  float relativeMax = normalizeInt(maxu16);
  relativeMin = std::max(relativeMin, 0.0f);
  relativeMax = std::min(relativeMax, 1.0f);
  if (relativeMin >= relativeMax) {
    LOG_ERROR << "Min value is greater than or equal to max value: " << minu16 << " >= " << maxu16
              << ", datarange=" << m_histogram.dataRange();
    return;
  }
  float window = relativeMax - relativeMin;
  float level = (relativeMax + relativeMin) / 2.0f;
  this->onSetWindowLevel(window, level);
}

void
GradientWidget::onSetIsovalue(float isovalue, float width)
{
  std::vector<LutControlPoint> points;
  float lowEnd = isovalue - width * 0.5f;
  float highEnd = isovalue + width * 0.5f;
  static const float epsilon = 0.00001f;
  points.push_back({ 0.0f, 0.0f });
  points.push_back({ lowEnd - epsilon, 0.0f });
  points.push_back({ lowEnd + epsilon, 1.0f });
  points.push_back({ highEnd - epsilon, 1.0f });
  points.push_back({ highEnd + epsilon, 0.0f });
  points.push_back({ 1.0f, 0.0f });
  m_editor->setControlPoints(points);
  emit gradientStopsChanged(vectorToGradientStops(points));
}
