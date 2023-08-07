#include "RangeWidget.h"

#include "Logging.h"

#include <QPainterPath>
#include <QSizePolicy>
#include <QtDebug>

constexpr int LABEL_SPACING = 4;
constexpr int OUTLINE_WIDTH = 1;

RangeWidget::RangeWidget(Qt::Orientation orientation, QWidget* parent)
  : QWidget(parent)
  , m_orientation(orientation)
  , m_handleWidth(18)
  , m_trackHeight(10)
  , m_handleHeight(24)
  , m_minBound(0)
  , m_maxBound(100)
  , m_firstValue(10)
  , m_secondValue(90)
  , m_firstHandlePressed(false)
  , m_secondHandlePressed(false)
  , m_firstHandleHovered(false)
  , m_secondHandleHovered(false)
  , m_trackHovered(false)
  , m_trackPressed(false)
  , m_firstHandleColor(style()->standardPalette().light().color())
  , m_secondHandleColor(style()->standardPalette().light().color())
  , m_trackFillColor(style()->standardPalette().mid().color())
  , m_trackOutlineColor(style()->standardPalette().midlight().color())
  , m_trackSelectionColor(QColor(0, 118, 246))
  , m_minSpinner()
  , m_maxSpinner()
  , m_layout()
{
  setLayout(&m_layout);
  // Set up first row (margin for where sliders will be drawn)
  m_layout.setRowMinimumHeight(0, m_handleHeight * 2);
  m_layout.addWidget(&m_minSpinner, 1, 0);
  m_layout.addWidget(&m_maxSpinner, 1, 2);
  // Set up stretching for the min and max spinners
  m_layout.setContentsMargins(0, 0, 0, 0); // keeps slider + spinners consistent
  m_layout.setColumnStretch(0, 1);
  m_layout.setColumnStretch(1, 3); // middle spacing column
  m_layout.setColumnStretch(2, 1);

  m_minSpinner.setAlignment(Qt::AlignCenter);
  m_maxSpinner.setAlignment(Qt::AlignCenter);
  // Set max very high so that large numbers can be typed in and clamped
  m_minSpinner.setMinimum(0);
  m_minSpinner.setMaximum(INT_MAX);
  m_maxSpinner.setMinimum(0);
  m_maxSpinner.setMaximum(INT_MAX);
  // Disable spinner handles
  m_minSpinner.setButtonSymbols(QAbstractSpinBox::NoButtons);
  m_maxSpinner.setButtonSymbols(QAbstractSpinBox::NoButtons);

  // Connect the spinners so they change the values + sliders
  QObject::connect(&m_minSpinner, QOverload<int>::of(&QSpinBox::valueChanged), [this](int v) { this->setMinValue(v); });
  QObject::connect(&m_maxSpinner, QOverload<int>::of(&QSpinBox::valueChanged), [this](int v) { this->setMaxValue(v); });

  setMouseTracking(true);
}

void
RangeWidget::paintEvent(QPaintEvent* event)
{
  int radius = 2;
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);
  int totalOutline = OUTLINE_WIDTH * 2;

  // First value handle rect
  QRectF rv1 = firstHandleRect();
  QColor c1(m_firstHandleColor);
  if (m_firstHandleHovered)
    c1 = c1.darker(125); // 80% of original brightness

  // Second value handle rect
  QRectF rv2 = secondHandleRect();
  QColor c2(m_secondHandleColor);
  if (m_secondHandleHovered)
    c2 = c2.darker(125);

  // Draw the track
  QRect r;
  if (m_orientation == Qt::Horizontal)
    r = QRect(0, std::floor((height() - m_trackHeight - totalOutline) / 2), width() - totalOutline, m_trackHeight);
  else
    r = QRect(std::floor((width() - m_trackHeight - totalOutline) / 2), 0, m_trackHeight, height() - 1);
  QPainterPath trackPath;
  trackPath.addRoundedRect(r.translated(0.5, 0.0), radius, radius);
  p.fillPath(trackPath, m_trackFillColor);

  // Color the selected range of the track
  QRectF rf(r);
  QPainterPath trackSelectionPath;
  if (m_orientation == Qt::Horizontal) {
    rf.setLeft(std::floor(rv1.right()));
    rf.setRight(std::floor(rv2.left()));
    rf.setTop(std::floor(rf.top()));
    rf.setBottom(std::floor(rf.bottom()));
  } else {
    rf.setTop(std::floor(rv1.bottom()));
    rf.setBottom(std::floor(rv2.top()));
    rf.setLeft(std::floor(rf.left()));
    rf.setRight(std::floor(rf.right()));
  }
  QColor trackSelectionColor(m_trackSelectionColor);
  if (m_trackHovered) {
    trackSelectionColor = trackSelectionColor.darker(125);
  }
  trackSelectionPath.addRect(rf);
  p.fillPath(trackSelectionPath, trackSelectionColor);
  m_trackRect = rf;

  // Draw handles
  p.setPen(style()->standardPalette().mid().color());

  p.setBrush(c1);
  QPainterPath rectPath1;

  // Translate by 0.5 so that rectangle aligns with pixel grid
  rectPath1.addRoundedRect(rv1.translated(0.5, 0.5), radius, radius);
  p.drawPath(rectPath1);

  QPainterPath rectPath2;
  p.setBrush(c2);
  rectPath2.addRoundedRect(rv2.translated(0.5, 0.5), radius, radius);
  p.drawPath(rectPath2);

  // Draw the knurling on the two handles
  QPainterPath handleKnurling;
  drawHandleKnurling(&handleKnurling, rv1);
  drawHandleKnurling(&handleKnurling, rv2);
  p.drawPath(handleKnurling);
}

/**
 * Adds three friction/knurling lines for the given handle to the path for drawing.
 */
void
RangeWidget::drawHandleKnurling(QPainterPath* path, QRectF handle, float widthRatio)
{
  float centerX = std::floor(handle.center().x()) + 0.5; // Center on pixel grid
  float centerY = std::floor(handle.center().y()) + 0.5;
  // Length of lines is 50% of handle height
  if (m_orientation == Qt::Horizontal) {
    float width = handle.top() - handle.bottom();
    float bottom = centerY + (width * widthRatio * 0.5); // Divide by two for centering
    float top = centerY - (width * widthRatio * 0.5);
    float offset = 2.0;
    path->moveTo(centerX, bottom);
    path->lineTo(centerX, top);
    path->moveTo(centerX + offset, bottom);
    path->lineTo(centerX + offset, top);
    path->moveTo(centerX - offset, bottom);
    path->lineTo(centerX - offset, top);
  } else {
    float left = handle.left() + m_handleHeight / 4.;
    float right = handle.right() - m_handleHeight / 4.;
    float centerY = std::floor(handle.center().y()) + 0.5;
    float offset = 2.0;
  }
}

qreal
RangeWidget::span(int w /* = -1 */) const
{
  int interval = qAbs(m_maxBound - m_minBound);

  if (m_orientation == Qt::Horizontal)
    return qreal(width() - (w == -1 ? m_handleWidth : w)) / qreal(interval);
  else
    return qreal(height() - (w == -1 ? m_handleWidth : w)) / qreal(interval);
}

QRectF
RangeWidget::firstHandleRect() const
{
  return handleRect(m_firstValue);
}

QRectF
RangeWidget::secondHandleRect() const
{
  return handleRect(m_secondValue);
}

QRectF
RangeWidget::handleRect(int value) const
{
  int totalOutline = OUTLINE_WIDTH * 2;
  qreal s = span();

  QRectF r;
  if (m_orientation == Qt::Horizontal) {
    r = QRectF(0,
               std::floor((height() - m_handleHeight - LABEL_SPACING - totalOutline) / 2),
               m_handleWidth - totalOutline,
               m_handleHeight);
    r.moveLeft(std::floor(s * (value - m_minBound)));
  } else {
    r = QRectF(std::floor((width() - m_handleHeight - LABEL_SPACING - totalOutline) / 2),
               0,
               m_handleHeight - totalOutline,
               m_handleWidth);
    r.moveTop(std::floor(s * (value - m_minBound)));
  }
  return r;
}

void
RangeWidget::mousePressEvent(QMouseEvent* event)
{
  if (event->buttons() & Qt::LeftButton) {
    m_secondHandlePressed = secondHandleRect().contains(event->pos());
    m_firstHandlePressed = !m_secondHandlePressed && firstHandleRect().contains(event->pos());
    m_trackPressed = !m_secondHandlePressed && !m_firstHandlePressed && m_trackRect.contains(event->pos());
    if (m_trackPressed) {
      m_trackPos = event->pos();
    }
    emit sliderPressed();
  }
}

void
RangeWidget::mouseMoveEvent(QMouseEvent* event)
{
  if (event->buttons() & Qt::LeftButton) {
    int interval = qAbs(m_maxBound - m_minBound);

    if (m_secondHandlePressed) {
      if (m_orientation == Qt::Horizontal)
        setSecondValue(event->pos().x() * interval / (width() - m_handleWidth));
      else
        setSecondValue(event->pos().y() * interval / (height() - m_handleWidth));
    } else if (m_firstHandlePressed) {
      if (m_orientation == Qt::Horizontal)
        setFirstValue(event->pos().x() * interval / (width() - m_handleWidth));
      else
        setFirstValue(event->pos().y() * interval / (height() - m_handleWidth));
    } else if (m_trackPressed) {
      int dx = event->pos().x() - m_trackPos.x();
      int dy = event->pos().y() - m_trackPos.y();
      int dvalue = 0;
      qreal s = span();
      if (m_orientation == Qt::Horizontal) {
        dvalue = (int)(dx / s + 0.5f);
      } else {
        dvalue = (int)(dy / s + 0.5f);
      }
      if (dvalue != 0) {
        m_trackPos = event->pos();
      }
      // LOG_DEBUG << "track delta " << dvalue;
      // Snap to min/max while maintaining the selection range if the change
      // would go past the bounds
      int range = valueRange();
      if (minValue() + (int)dvalue < m_minBound) {
        // Snap to min
        setMaxValue(m_minBound + range);
        setMinValue(m_minBound);
      } else if (maxValue() + (int)dvalue > m_maxBound) {
        // Snap to max
        setMinValue(m_maxBound - range);
        setMaxValue(m_maxBound);
      } else {
        setFirstValue(m_firstValue + (int)dvalue);
        setSecondValue(m_secondValue + (int)dvalue);
      }
    }
  }

  updateHoverFlags(event);
}

void
RangeWidget::mouseReleaseEvent(QMouseEvent* event)
{
  if (m_firstHandlePressed || m_secondHandlePressed || m_trackPressed)
    emit sliderReleased();

  m_firstHandlePressed = false;
  m_secondHandlePressed = false;
  m_trackPressed = false;
  // Reset hovering
  updateHoverFlags(event);
}

void
RangeWidget::updateHoverFlags(QMouseEvent* event)
{
  QRectF rv2 = secondHandleRect();
  QRectF rv1 = firstHandleRect();
  m_secondHandleHovered = m_secondHandlePressed || (!m_firstHandlePressed && rv2.contains(event->pos()));
  m_firstHandleHovered = m_firstHandlePressed || (!m_secondHandleHovered && rv1.contains(event->pos()));
  m_trackHovered = m_trackPressed || m_trackRect.contains(event->pos());
  update(rv2.toRect());
  update(rv1.toRect());
  update(m_trackRect.toRect());
}

QSize
RangeWidget::minimumSizeHint() const
{
  return QSize(m_handleHeight * 2 + LABEL_SPACING, m_handleHeight * 2 + LABEL_SPACING);
}

void
RangeWidget::setMinValue(int value, bool blockSignals)
{
  if (m_firstValue < m_secondValue) {
    setFirstValue(value, blockSignals);
  } else {
    setSecondValue(value, blockSignals);
  }
}

void
RangeWidget::setMaxValue(int value, bool blockSignals)
{
  if (m_firstValue > m_secondValue) {
    setFirstValue(value, blockSignals);
  } else {
    setSecondValue(value, blockSignals);
  }
}

void
RangeWidget::updateSpinners()
{
  m_minSpinner.blockSignals(true);
  m_maxSpinner.blockSignals(true);
  m_minSpinner.setValue(minValue());
  m_maxSpinner.setValue(maxValue());
  m_minSpinner.blockSignals(false);
  m_maxSpinner.blockSignals(false);
}

void
RangeWidget::setSecondValue(int secondValue, bool blockSignals)
{
  if (secondValue > m_maxBound)
    secondValue = m_maxBound;

  if (secondValue < m_minBound)
    secondValue = m_minBound;

  // Compare against last value and determine whether firstValue is the min or max
  bool secondValueIsMin = m_secondValue < m_firstValue;
  bool didMinMaxSwap = secondValueIsMin != (secondValue < m_firstValue);
  m_secondValue = secondValue;

  if (!blockSignals) {
    // Broadcast new min and/or max.
    if (didMinMaxSwap || secondValueIsMin) {
      emit minValueChanged(minValue());
    }
    if (didMinMaxSwap || !secondValueIsMin) {
      emit maxValueChanged(maxValue());
    }
  }

  updateSpinners();
  update();
}

void
RangeWidget::setFirstValue(int firstValue, bool blockSignals)
{
  if (firstValue > m_maxBound)
    firstValue = m_maxBound;

  if (firstValue < m_minBound)
    firstValue = m_minBound;

  // Compare against last value and determine whether firstValue is the min or max
  bool firstValueIsMin = m_firstValue < m_secondValue;
  bool didMinMaxSwap = firstValueIsMin != (firstValue < m_secondValue);
  m_firstValue = firstValue;

  if (!blockSignals) {
    // Broadcast new min and/or max.
    if (didMinMaxSwap || firstValueIsMin) {
      emit minValueChanged(minValue());
    }
    if (didMinMaxSwap || !firstValueIsMin) {
      emit maxValueChanged(maxValue());
    }
  }

  updateSpinners();
  update();
}

void
RangeWidget::setMaxBound(int max, bool blockSignals)
{
  if (max >= minBound())
    m_maxBound = max;
  else {
    int oldMin = minBound();
    m_maxBound = oldMin;
    m_minBound = max;
  }

  if (minValue() > maxBound())
    setMinValue(maxBound(), blockSignals);

  if (maxValue() > maxBound())
    setMaxValue(maxBound(), blockSignals);

  updateSpinners();
  update();

  if (!blockSignals) {
    emit rangeChanged(minBound(), maxBound());
  }
}

void
RangeWidget::setBounds(int min, int max, bool blockSignals)
{
  setMinBound(min, blockSignals);
  setMaxBound(max, blockSignals);
}

void
RangeWidget::setMinBound(int min, bool blockSignals)
{
  if (min <= maxBound())
    m_minBound = min;
  else {
    int oldMax = maxBound();
    m_minBound = oldMax;
    m_maxBound = min;
  }

  if (minValue() < minBound())
    setMinValue(minBound(), blockSignals);

  if (maxValue() < minBound())
    setMaxValue(minBound(), blockSignals);

  updateSpinners();
  update();

  if (!blockSignals) {
    emit rangeChanged(minBound(), maxBound());
  }
}

void
RangeWidget::setOrientation(Qt::Orientation orientation)
{
  if (m_orientation == orientation)
    return;

  m_orientation = orientation;
  updateSpinners();
  update();
}
