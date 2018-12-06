#pragma once

#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QInputDialog>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>

class QColorPushButton : public QPushButton
{
  Q_OBJECT

public:
  QColorPushButton(QWidget* pParent = NULL);

  virtual QSize sizeHint() const;
  virtual void paintEvent(QPaintEvent* pPaintEvent);
  virtual void mousePressEvent(QMouseEvent* pEvent);

  int GetMargin(void) const;
  void SetMargin(const int& Margin);
  int GetRadius(void) const;
  void SetRadius(const int& Radius);
  QColor GetColor(void) const;
  void SetColor(const QColor& Color, bool BlockSignals = false);

private slots:
  void OnCurrentColorChanged(const QColor& Color);

signals:
  void currentColorChanged(const QColor&);

private:
  int m_Margin;
  int m_Radius;
  QColor m_Color;
};

class QColorSelector : public QFrame
{
  Q_OBJECT

public:
  QColorSelector(QWidget* pParent = NULL);

  //	virtual QSize sizeHint() const;

  QColor GetColor(void) const;
  void SetColor(const QColor& Color, bool BlockSignals = false);

private slots:
  void OnCurrentColorChanged(const QColor& Color);

signals:
  void currentColorChanged(const QColor&);

private:
  QGridLayout m_MainLayout;
  QColorPushButton m_ColorButton;
  QComboBox m_ColorCombo;
};

class QDoubleSlider : public QSlider
{
  Q_OBJECT

public:
  QDoubleSlider(QWidget* pParent = NULL);

  void setRange(double Min, double Max);
  void setMinimum(double Min);
  double minimum() const;
  void setMaximum(double Max);
  double maximum() const;
  double value() const;

public slots:
  void setValue(int value);
  void setValue(double Value, bool BlockSignals = false);

private slots:

signals:
  void valueChanged(double Value);
  void rangeChanged(double Min, double Max);

private:
  double m_Multiplier;
};

class QDoubleSpinner : public QDoubleSpinBox
{
  Q_OBJECT

public:
  QDoubleSpinner(QWidget* pParent = NULL);
  ;

  virtual QSize sizeHint() const;
  void setValue(double Value, bool BlockSignals = false);
};

class QInputDialogEx : public QInputDialog
{
public:
  QInputDialogEx(QWidget* pParent = NULL, Qt::WindowFlags Flags = 0);

  virtual QSize sizeHint() const;
};

class QNumericSlider : public QWidget
{
  Q_OBJECT
public:
  QNumericSlider(QWidget* pParent = NULL);

  double value(void) const;
  void setValue(double value, bool BlockSignals = false);
  void setRange(double rmin, double rmax);
  void setDecimals(int decimals);
  void setSuffix(const QString&);

private slots:
  void OnValueChanged(double value);

signals:
  void valueChanged(double value);

private:
  QGridLayout m_layout;
  QDoubleSpinner m_spinner;
  QDoubleSlider m_slider;
};
