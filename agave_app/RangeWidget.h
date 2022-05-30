#ifndef RANGEWIDGET_H
#define RANGEWIDGET_H

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

  int m_minimum;
  int m_maximum;

  int m_firstValue;
  int m_secondValue;

  bool m_firstHandlePressed;
  bool m_secondHandlePressed;

  bool m_firstHandleHovered;
  bool m_secondHandleHovered;

  QColor m_firstHandleColor;
  QColor m_secondHandleColor;

protected:
  void paintEvent(QPaintEvent* event);
  void mousePressEvent(QMouseEvent* event);
  void mouseMoveEvent(QMouseEvent* event);
  void mouseReleaseEvent(QMouseEvent* event);

  QRectF firstHandleRect() const;
  QRectF secondHandleRect() const;
  QRectF handleRect(int value) const;
  QRectF firstTextRect(QPainter& p) const;
  QRectF secondTextRect(QPainter& p) const;
  QRectF textRect(int value, QPainter& p) const;
  qreal span(int w=-1) const;

public:
  RangeWidget(Qt::Orientation orientation = Qt::Vertical, QWidget* parent = nullptr);

  QSize minimumSizeHint() const;

  inline int firstValue() const { return m_firstValue; }
  inline int secondValue() const { return m_secondValue; }
  inline int minimum() const { return m_minimum; }
  inline int maximum() const { return m_maximum; }
  inline Qt::Orientation orientation() const { return m_orientation; }
  inline int interval() const { return secondValue() - firstValue(); }
  inline unsigned int absInterval() const { return qAbs(interval()); }

signals:
  void firstValueChanged(int firstValue);
  void secondValueChanged(int secondValue);
  void rangeChanged(int min, int max);
  void sliderPressed();
  void sliderReleased();

public slots:
  void setFirstValue(int firstValue);
  void setSecondValue(int secondValue);
  void setMinimum(int min);
  void setMaximum(int max);
  void setRange(int min, int max);
  void setOrientation(Qt::Orientation orientation);
};

#endif // RANGEWIDGET_H
