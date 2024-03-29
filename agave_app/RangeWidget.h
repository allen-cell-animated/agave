#ifndef RANGEWIDGET_H
#define RANGEWIDGET_H

#include <Controls.h>
#include <QMouseEvent>
#include <QPainter>
#include <QStyle>
#include <QWidget>

class RangeWidget : public QWidget
{
  Q_OBJECT
private:
  Q_DISABLE_COPY(RangeWidget)

  Qt::Orientation m_orientation;

  int m_handleWidth;
  int m_handleHeight;

  int m_trackHeight;

  int m_minBound;
  int m_maxBound;

  int m_firstValue;
  int m_secondValue;

  bool m_firstHandlePressed;
  bool m_secondHandlePressed;

  bool m_firstHandleHovered;
  bool m_secondHandleHovered;

  bool m_trackHovered;
  bool m_trackPressed;

  QColor m_firstHandleColor;
  QColor m_secondHandleColor;
  QColor m_trackSelectionColor;
  QColor m_trackOutlineColor;
  QColor m_trackFillColor;

  QRectF m_trackRect;
  QPoint m_trackPos;
  MySpinBoxWithEnter m_minSpinner;
  MySpinBoxWithEnter m_maxSpinner;
  QGridLayout m_layout;

protected:
  void paintEvent(QPaintEvent* event);
  void mousePressEvent(QMouseEvent* event);
  void mouseMoveEvent(QMouseEvent* event);
  void mouseReleaseEvent(QMouseEvent* event);
  void updateSpinners();
  void updateHoverFlags(QMouseEvent* event);

  QRectF firstHandleRect() const;
  QRectF secondHandleRect() const;
  QRectF handleRect(int value) const;
  QRectF firstTextRect(QPainter& p) const;
  QRectF secondTextRect(QPainter& p) const;
  QRectF textRect(int value, QPainter& p) const;
  qreal span(int w = -1) const;
  void makeHandleGripLines(QPainterPath* path, QRectF handle, float widthRatio = 0.4);

public:
  RangeWidget(Qt::Orientation orientation = Qt::Vertical, QWidget* parent = nullptr);

  QSize minimumSizeHint() const;

  inline int minValue() const { return std::min(m_firstValue, m_secondValue); }
  inline float minValuePercent() const { return (float)(minValue() - minBound()) / (float)boundsRange(); }
  inline int maxValue() const { return std::max(m_firstValue, m_secondValue); }
  inline float maxValuePercent() const { return (float)(maxValue() - minBound()) / (float)boundsRange(); }
  inline int minBound() const { return m_minBound; }
  inline int maxBound() const { return m_maxBound; }
  inline int boundsRange() const { return maxBound() - minBound(); }
  inline Qt::Orientation orientation() const { return m_orientation; }
  inline int valueRange() const { return maxValue() - minValue(); }
  inline unsigned int valueRangeAbs() const { return qAbs(valueRange()); }

signals:
  void minValueChanged(int firstValue);
  void maxValueChanged(int secondValue);
  void rangeChanged(int min, int max);
  void sliderPressed();
  void sliderReleased();

public slots:
  void setFirstValue(int firstValue, bool blockSignals = false);
  void setSecondValue(int secondValue, bool blockSignals = false);
  void setMinValue(int value, bool blockSignals = false);
  void setMaxValue(int value, bool blockSignals = false);
  void setMinBound(int min, bool blockSignals = false);
  void setMaxBound(int max, bool blockSignals = false);
  void setBounds(int min, int max, bool blockSignals = false);
  void setOrientation(Qt::Orientation orientation);
};

#endif // RANGEWIDGET_H
