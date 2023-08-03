#include "RangeWidget.h"

#include "Logging.h"

#include <QSizePolicy>
#include <QtDebug>

constexpr int LABEL_SPACING = 4;
constexpr int OUTLINE_WIDTH = 1;

RangeWidget::RangeWidget(Qt::Orientation orientation, QWidget* parent)
  : QWidget(parent)
  , m_orientation(orientation)
  , m_handleWidth(12)
  , m_trackHeight(5)
  , m_handleHeight(20)
  , m_minimum(0)
  , m_maximum(100)
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
  , m_trackFillColor(style()->standardPalette().midlight().color())
  , m_trackOutlineColor(style()->standardPalette().window().color())
  , m_trackRangeColor(style()->standardPalette().dark().color())
  , m_minSpinner()
  , m_maxSpinner()
  , m_layout()
{
  setLayout(&m_layout);
  // Set up first row (margin for where sliders will be drawn)
  m_layout.setRowMinimumHeight(0, m_handleHeight * 2);
  m_layout.addWidget(&m_minSpinner, 1, 0);
  m_layout.addWidget(&m_maxSpinner, 1, 2);
  m_layout.setContentsMargins(0, 0, 0, 0);
  m_layout.setColumnStretch(0, 1);
  m_layout.setColumnStretch(1, 3); // middle spacing column
  m_layout.setColumnStretch(2, 1);

  m_minSpinner.setAlignment(Qt::AlignCenter);
  m_maxSpinner.setAlignment(Qt::AlignCenter);
  m_minSpinner.setMinimum(0);
  m_minSpinner.setMaximum(INT_MAX);
  m_maxSpinner.setMinimum(0);
  m_maxSpinner.setMaximum(INT_MAX);

  // Connect the spinners so they change the values + sliders
  QObject::connect(&m_minSpinner, QOverload<int>::of(&QSpinBox::valueChanged), [this](int v) { this->setMinValue(v); });
  QObject::connect(&m_maxSpinner, QOverload<int>::of(&QSpinBox::valueChanged), [this](int v) { this->setMaxValue(v); });

  // Add a second row
  setMouseTracking(true);
}

void
RangeWidget::paintEvent(QPaintEvent* event)
{
  QPainter p(this);
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

  // Track
  QRect r;
  if (m_orientation == Qt::Horizontal)
    r = QRect(0, (height() - m_trackHeight - LABEL_SPACING - totalOutline) / 2, width() - totalOutline, m_trackHeight);
  else
    r = QRect((width() - m_trackHeight - LABEL_SPACING) / 2, 0, m_trackHeight, height() - 1);
  p.fillRect(r, m_trackFillColor);
  p.setPen(m_trackOutlineColor);
  p.drawRect(r);

  // Track range
  QRectF rf(r);
  if (m_orientation == Qt::Horizontal) {
    rf.setLeft(rv1.right());
    rf.setRight(rv2.left());
    rf.setBottom(rf.bottom() + 1);
  } else {
    rf.setTop(rv1.bottom());
    rf.setBottom(rv2.top());
    rf.setRight(rf.right() + 1);
  }
  p.fillRect(rf, m_trackRangeColor);

  // Handles
  m_trackRect = rf;
  p.setPen(style()->standardPalette().mid().color());
  p.fillRect(rv1, c1);
  p.drawRect(rv1);
  p.fillRect(rv2, c2);
  p.drawRect(rv2);
  p.setPen(style()->standardPalette().text().color());
}

qreal
RangeWidget::span(int w /* = -1 */) const
{
  int interval = qAbs(m_maximum - m_minimum);

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
    r = QRectF(
      0, (height() - m_handleHeight - LABEL_SPACING - totalOutline) / 2, m_handleWidth - totalOutline, m_handleHeight);
    r.moveLeft(s * (value - m_minimum));
  } else {
    r = QRectF(
      (width() - m_handleHeight - LABEL_SPACING - totalOutline) / 2, 0, m_handleHeight - totalOutline, m_handleWidth);
    r.moveTop(s * (value - m_minimum));
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
    int interval = qAbs(m_maximum - m_minimum);

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
      if (m_firstValue + (int)dvalue >= m_minimum && m_secondValue + (int)dvalue <= m_maximum) {
        setFirstValue(m_firstValue + (int)dvalue);
        setSecondValue(m_secondValue + (int)dvalue);
      }
    }
  }

  QRectF rv2 = secondHandleRect();
  QRectF rv1 = firstHandleRect();
  m_secondHandleHovered = m_secondHandlePressed || (!m_firstHandlePressed && rv2.contains(event->pos()));
  m_firstHandleHovered = m_firstHandlePressed || (!m_secondHandleHovered && rv1.contains(event->pos()));
  m_trackHovered = m_trackPressed || (m_trackRect.contains(event->pos()));
  update(rv2.toRect());
  update(rv1.toRect());
  update(m_trackRect.toRect());
}

void
RangeWidget::mouseReleaseEvent(QMouseEvent* event)
{
  if (m_firstHandlePressed || m_secondHandlePressed)
    emit sliderReleased();

  m_firstHandlePressed = false;
  m_secondHandlePressed = false;
}

QSize
RangeWidget::minimumSizeHint() const
{
  return QSize(m_handleHeight * 2 + LABEL_SPACING, m_handleHeight * 2 + LABEL_SPACING);
}

void
RangeWidget::setMinValue(int value, bool blockSignals)
{
  LOG_INFO << value;
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
  m_minSpinner.setValue(valueMin());
  m_maxSpinner.setValue(valueMax());
  m_minSpinner.blockSignals(false);
  m_maxSpinner.blockSignals(false);
}

void
RangeWidget::setSecondValue(int secondValue, bool blockSignals)
{
  if (secondValue > m_maximum)
    secondValue = m_maximum;

  if (secondValue < m_minimum)
    secondValue = m_minimum;

  // Compare against last value and determine whether firstValue is the min or max
  bool secondValueIsMin = m_secondValue < m_firstValue;
  bool didMinMaxSwap = secondValueIsMin != (secondValue < m_firstValue);
  m_secondValue = secondValue;

  if (!blockSignals) {
    // Broadcast new min and/or max.
    if (didMinMaxSwap || secondValueIsMin) {
      emit minValueChanged(valueMin());
    }
    if (didMinMaxSwap || !secondValueIsMin) {
      emit maxValueChanged(valueMax());
    }
  }

  updateSpinners();
  update();
}

void
RangeWidget::setFirstValue(int firstValue, bool blockSignals)
{
  if (firstValue > m_maximum)
    firstValue = m_maximum;

  if (firstValue < m_minimum)
    firstValue = m_minimum;

  // Compare against last value and determine whether firstValue is the min or max
  bool firstValueIsMin = m_firstValue < m_secondValue;
  bool didMinMaxSwap = firstValueIsMin != (firstValue < m_secondValue);
  m_firstValue = firstValue;

  if (!blockSignals) {
    // Broadcast new min and/or max.
    if (didMinMaxSwap || firstValueIsMin) {
      emit minValueChanged(valueMin());
    }
    if (didMinMaxSwap || !firstValueIsMin) {
      emit maxValueChanged(valueMax());
    }
  }

  updateSpinners();
  update();
}

void
RangeWidget::setBoundsMax(int max, bool blockSignals)
{
  if (max >= boundsMin())
    m_maximum = max;
  else {
    int oldMin = boundsMin();
    m_maximum = oldMin;
    m_minimum = max;
  }

  if (valueMin() > boundsMax())
    setMinValue(boundsMax(), blockSignals);

  if (valueMax() > boundsMax())
    setMaxValue(boundsMax(), blockSignals);

  updateSpinners();
  update();

  if (!blockSignals) {
    emit rangeChanged(boundsMin(), boundsMax());
  }
}

void
RangeWidget::setBounds(int min, int max, bool blockSignals)
{
  setBoundsMin(min, blockSignals);
  setBoundsMax(max, blockSignals);
}

void
RangeWidget::setBoundsMin(int min, bool blockSignals)
{
  if (min <= boundsMax())
    m_minimum = min;
  else {
    int oldMax = boundsMax();
    m_minimum = oldMax;
    m_maximum = min;
  }

  if (valueMin() < boundsMin())
    setMinValue(boundsMin(), blockSignals);

  if (valueMax() < boundsMin())
    setMaxValue(boundsMin(), blockSignals);

  updateSpinners();
  update();

  if (!blockSignals) {
    emit rangeChanged(boundsMin(), boundsMax());
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
