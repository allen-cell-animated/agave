#include "RangeWidget.h"

#include "Logging.h"

#include <QtDebug>

RangeWidget::RangeWidget(Qt::Orientation orientation, QWidget* parent)
  : QWidget(parent)
  , m_orientation(orientation)
  , m_handleWidth(8)
  , m_handleHeight(20)
  , m_minimum(0)
  , m_maximum(100)
  , m_firstValue(10)
  , m_secondValue(90)
  , m_firstHandlePressed(false)
  , m_secondHandlePressed(false)
  , m_trackHovered(false)
  , m_trackPressed(false)
  , m_firstHandleColor(style()->standardPalette().highlight().color())
  , m_secondHandleColor(style()->standardPalette().highlight().color())
{
  setMouseTracking(true);
}

void
RangeWidget::paintEvent(QPaintEvent* event)
{
  QPainter p(this);

  // First value handle rect
  QRectF rt1 = firstTextRect(p);
  QRectF rv1 = firstHandleRect();
  QColor c1(m_firstHandleColor);
  if (m_firstHandleHovered)
    c1 = c1.darker();

  // Second value handle rect
  QRectF rt2 = secondTextRect(p);
  QRectF rv2 = secondHandleRect();
  QColor c2(m_secondHandleColor);
  if (m_secondHandleHovered)
    c2 = c2.darker();

  // Background
  QRect r;
  if (m_orientation == Qt::Horizontal)
    r = QRect(0, (height() - m_handleWidth) / 2, width() - 1, m_handleWidth);
  else
    r = QRect((width() - m_handleWidth) / 2, 0, m_handleWidth, height() - 1);
  p.drawRect(r);

  // Handles
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
  p.fillRect(rf, QColor(Qt::green).darker(150));
  m_trackRect = rf;
  p.fillRect(rv1, c1);
  p.fillRect(rv2, c2);
  p.drawText(rt1, Qt::AlignmentFlag::AlignCenter, QString::number(m_firstValue));
  p.drawText(rt2, Qt::AlignmentFlag::AlignCenter, QString::number(m_secondValue));
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
  qreal s = span();

  QRectF r;
  if (m_orientation == Qt::Horizontal) {
    r = QRectF(0, (height() - m_handleHeight) / 2, m_handleWidth, m_handleHeight);
    r.moveLeft(s * (value - m_minimum));
  } else {
    r = QRectF((width() - m_handleHeight) / 2, 0, m_handleHeight, m_handleWidth);
    r.moveTop(s * (value - m_minimum));
  }
  return r;
}

QRectF
RangeWidget::firstTextRect(QPainter& p) const
{
  return textRect(m_firstValue, p);
}

QRectF
RangeWidget::secondTextRect(QPainter& p) const
{
  return textRect(m_secondValue, p);
}

QRectF
RangeWidget::textRect(int value, QPainter& p) const
{
  QRect rt;
  rt = p.boundingRect(rt, Qt::AlignCenter, QString::number(value));

  qreal s = span(rt.width());

  QRectF r;
  if (m_orientation == Qt::Horizontal) {
    r = rt;
    r.moveLeft(s * (value - m_minimum));
    r.moveTop(height() - rt.height());
  } else {
    r = rt;
    r.moveLeft(width() - rt.width());
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
      m_trackPos = event->pos();
      qreal dvalue = 0.0;
      qreal s = span();
      if (m_orientation == Qt::Horizontal) {
        dvalue = dx / s;
      } else {
        dvalue = dy / s;
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
  return QSize(m_handleHeight * 2, m_handleHeight * 2);
}

void
RangeWidget::setSecondValue(int secondValue)
{
  if (secondValue > m_maximum)
    secondValue = m_maximum;

  if (secondValue < m_minimum)
    secondValue = m_minimum;

  m_secondValue = secondValue;
  emit secondValueChanged(m_secondValue);

  update();
}

void
RangeWidget::setFirstValue(int firstValue)
{
  if (firstValue > m_maximum)
    firstValue = m_maximum;

  if (firstValue < m_minimum)
    firstValue = m_minimum;

  m_firstValue = firstValue;
  emit firstValueChanged(m_firstValue);

  update();
}

void
RangeWidget::setMaximum(int max)
{
  if (max >= minimum())
    m_maximum = max;
  else {
    int oldMin = minimum();
    m_maximum = oldMin;
    m_minimum = max;
  }

  update();

  if (firstValue() > maximum())
    setFirstValue(maximum());

  if (secondValue() > maximum())
    setSecondValue(maximum());

  emit rangeChanged(minimum(), maximum());
}

void
RangeWidget::setRange(int min, int max)
{
  setMinimum(min);
  setMaximum(max);
}

void
RangeWidget::setMinimum(int min)
{
  if (min <= maximum())
    m_minimum = min;
  else {
    int oldMax = maximum();
    m_minimum = oldMax;
    m_maximum = min;
  }

  update();

  if (firstValue() < minimum())
    setFirstValue(minimum());

  if (secondValue() < minimum())
    setSecondValue(minimum());

  emit rangeChanged(minimum(), maximum());
}

void
RangeWidget::setOrientation(Qt::Orientation orientation)
{
  if (m_orientation == orientation)
    return;

  m_orientation = orientation;
  update();
}
