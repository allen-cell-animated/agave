#include "Controls.h"

#include <QColorDialog>
#include <QPaintEvent>
#include <QPainter>
#include <QPainterPath>
#include <QtGlobal>

QColorPushButton::QColorPushButton(QWidget* pParent)
  : QPushButton(pParent)
  , m_Margin(5)
  , m_Radius(4)
  , m_Color(Qt::gray)
{
  setText("");
}

void
QColorPushButton::paintEvent(QPaintEvent* pPaintEvent)
{
  setText("");

  // QPushButton::paintEvent(pPaintEvent);

  QPainter painter(this);

  // Get button rectangle
  QRect colorRectangle = pPaintEvent->rect();

  // Deflate it
  colorRectangle.adjust(m_Margin, m_Margin, -m_Margin, -m_Margin);

  // Use anti aliasing
  painter.setRenderHint(QPainter::Antialiasing);

  QPainterPath path;
  path.addRoundedRect(colorRectangle, m_Radius, m_Radius, Qt::AbsoluteSize);
  QPen pen(isEnabled() ? QColor(25, 25, 25) : Qt::darkGray, 0.5);
  painter.setPen(pen);
  painter.fillPath(path, isEnabled() ? m_Color : Qt::lightGray);
  painter.drawPath(path);
}

void
QColorPushButton::mousePressEvent(QMouseEvent* pEvent)
{
  QColor lastColor = m_Color;
  QColorDialog colorDialog;

  connect(&colorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));

#ifdef __linux__
  colorDialog.setOption(QColorDialog::DontUseNativeDialog, true);
#endif
  colorDialog.setCurrentColor(m_Color);
  int result = colorDialog.exec();
  if (result == QDialog::Rejected) {
    OnCurrentColorChanged(lastColor);
  }

  disconnect(
    &colorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));
}

int
QColorPushButton::GetMargin(void) const
{
  return m_Margin;
}

void
QColorPushButton::SetMargin(const int& margin)
{
  m_Margin = margin;
  update();
}

int
QColorPushButton::GetRadius(void) const
{
  return m_Radius;
}

void
QColorPushButton::SetRadius(const int& radius)
{
  m_Radius = radius;
  update();
}

QColor
QColorPushButton::GetColor(void) const
{
  return m_Color;
}

void
QColorPushButton::SetColor(const QColor& color, bool blockSignals)
{
  this->blockSignals(blockSignals);

  QPalette pal = palette();
  pal.setColor(QPalette::Button, color);
  setPalette(pal);

  m_Color = color;
  update();
  if (!blockSignals) {
    emit currentColorChanged(m_Color);
  }

  this->blockSignals(false);
}

void
QColorPushButton::OnCurrentColorChanged(const QColor& color)
{
  SetColor(color, true);

  emit currentColorChanged(m_Color);
}

QColorSelector::QColorSelector(QWidget* pParent /*= NULL*/)
  : QFrame(pParent)
  , m_ColorButton()
  , m_ColorCombo()
{
  setLayout(&m_MainLayout);

  m_MainLayout.addWidget(&m_ColorButton, 0, 0, Qt::AlignLeft);
  //	m_MainLayout.addWidget(&m_ColorCombo, 0, 1);

  m_MainLayout.setContentsMargins(0, 0, 0, 0);

  m_ColorButton.setFixedWidth(30);

  QObject::connect(
    &m_ColorButton, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));
}

QColor
QColorSelector::GetColor(void) const
{
  return m_ColorButton.GetColor();
}

void
QColorSelector::SetColor(const QColor& color, bool blockSignals /*= false*/)
{
  m_ColorButton.SetColor(color, blockSignals);
}

void
QColorSelector::OnCurrentColorChanged(const QColor& color)
{
  emit currentColorChanged(color);
}

QDoubleSlider::QDoubleSlider(QWidget* pParent /*= NULL*/)
  : QSlider(pParent)
  , m_Multiplier(10000.0)
{
  connect(this, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));

  setSingleStep(1);

  setOrientation(Qt::Horizontal);
  setFocusPolicy(Qt::StrongFocus);
}

void
QDoubleSlider::setValue(int value)
{
  emit valueChanged((double)value / m_Multiplier);
}

void
QDoubleSlider::setValue(double value, bool blockSignals)
{
  QSlider::blockSignals(blockSignals);

  QSlider::setValue(value * m_Multiplier);

  if (!blockSignals)
    emit valueChanged(value);

  QSlider::blockSignals(false);
}

void
QDoubleSlider::setRange(double rmin, double rmax)
{
  QSlider::setRange(rmin * m_Multiplier, rmax * m_Multiplier);

  emit rangeChanged(rmin, rmax);
}

void
QDoubleSlider::setMinimum(double dmin)
{
  QSlider::setMinimum(dmin * m_Multiplier);

  emit rangeChanged(minimum(), maximum());
}

double
QDoubleSlider::minimum() const
{
  return QSlider::minimum() / m_Multiplier;
}

void
QDoubleSlider::setMaximum(double dmax)
{
  QSlider::setMaximum(dmax * m_Multiplier);

  emit rangeChanged(minimum(), maximum());
}

double
QDoubleSlider::maximum() const
{
  return QSlider::maximum() / m_Multiplier;
}

double
QDoubleSlider::value() const
{
  int value = QSlider::value();
  return sliderPositionToValue(value);
}

double
QDoubleSlider::sliderPositionToValue(int pos) const
{
  return (double)pos / m_Multiplier;
}

void
QDoubleSlider::wheelEvent(QWheelEvent* event)
{
  if (!hasFocus()) {
    event->ignore();
    return;
  }
  QSlider::wheelEvent(event);
}

QSize
QDoubleSpinner::sizeHint() const
{
  return QSize(50, 20);
}

QDoubleSpinner::QDoubleSpinner(QWidget* pParent /*= NULL*/)
  : QDoubleSpinBox(pParent)
{
  setKeyboardTracking(false);
}

void
QDoubleSpinner::setValue(double value, bool blockSignals)
{
  this->blockSignals(blockSignals);

  QDoubleSpinBox::setValue(value);

  this->blockSignals(false);
}

void
QDoubleSpinner::wheelEvent(QWheelEvent* event)
{
  if (!hasFocus()) {
    event->ignore();
    return;
  }
  QDoubleSpinBox::wheelEvent(event);
}

QNumericSlider::QNumericSlider(QWidget* pParent /*= NULL*/)
  : QWidget(pParent)
  , m_slider()
  , m_spinner()
{
  setLayout(&m_layout);

  m_slider.setOrientation(Qt::Horizontal);
  m_slider.setFocusPolicy(Qt::StrongFocus);
  m_spinner.setDecimals(4);
  m_spinner.setFocusPolicy(Qt::StrongFocus);

  // entire control is one single row.
  // slider is 3/4, spinner is 1/4 of the width
  const int sliderratio = 4;
  m_layout.addWidget(&m_slider, 0, 0, 1, sliderratio - 1);
  m_layout.addWidget(&m_spinner, 0, sliderratio - 1, 1, 1);

  m_layout.setContentsMargins(0, 0, 0, 0);

  // keep slider and spinner in sync
  QObject::connect(&m_slider, &QDoubleSlider::valueChanged, [this](double v) {
    this->m_spinner.blockSignals(true);
    this->m_spinner.setValue(v, true);
    this->m_spinner.blockSignals(false);
  });

  QObject::connect(&m_spinner, QOverload<double>::of(&QDoubleSpinner::valueChanged), [this](double v) {
    // Disabling the spinner only because the valueChanged handler might do a long operation when the spinner is
    // clicked.
    //
    // If the operation is "too long" then the mouserelease of the spinner will kick in after a certain timeout
    // and trigger one more increment.
    //
    // If the owner has disabled tracking, that's a good bet that the increments are
    // expected to be "too long".
    if (!this->m_slider.hasTracking()) {
      this->m_spinner.setEnabled(false);
    }
    this->m_slider.setValue(v);
    if (!this->m_slider.hasTracking()) {
      this->m_spinner.setEnabled(true);
    }
  });

  // only slider will emit the value...
  QObject::connect(&m_slider, SIGNAL(valueChanged(double)), this, SLOT(OnValueChanged(double)));
}

void
QNumericSlider::OnValueChanged(double value)
{
  emit valueChanged(value);
}

double
QNumericSlider::value(void) const
{
  return m_spinner.value();
}

void
QNumericSlider::setValue(double value, bool blockSignals)
{
  // only forward the blocksignals flag for one of the two child controls.
  // the other will always block signalling
  m_spinner.setValue(value, true);
  m_slider.setValue(value, blockSignals);
}

void
QNumericSlider::setRange(double rmin, double rmax)
{
  m_slider.setRange(rmin, rmax);
  m_spinner.setRange(rmin, rmax);
}

void
QNumericSlider::setSingleStep(double val)
{
  m_spinner.setSingleStep(val);
}

void
QNumericSlider::setDecimals(int decimals)
{
  m_spinner.setDecimals(decimals);
}

void
QNumericSlider::setSuffix(const QString& s)
{
  m_spinner.setSuffix(s);
}

void
QNumericSlider::setTracking(bool enabled)
{
  m_slider.setTracking(enabled);
}

QIntSlider::QIntSlider(QWidget* pParent /*= NULL*/)
  : QWidget(pParent)
  , m_slider()
  , m_spinner()
{
  setLayout(&m_layout);

  m_slider.setOrientation(Qt::Horizontal);
  m_slider.setFocusPolicy(Qt::NoFocus);

  m_spinner.setKeyboardTracking(false);

  // entire control is one single row.
  // slider is 3/4, spinner is 1/4 of the width
  const int sliderratio = 4;
  m_layout.addWidget(&m_slider, 0, 0, 1, sliderratio - 1);
  m_layout.addWidget(&m_spinner, 0, sliderratio - 1, 1, 1);

  m_layout.setContentsMargins(0, 0, 0, 0);

  // keep slider and spinner in sync
  QObject::connect(&m_slider, QOverload<int>::of(&QSlider::sliderMoved), [this](int v) {
    this->m_spinner.blockSignals(true);
    this->m_spinner.setValue(v);
    this->m_spinner.blockSignals(false);
  });
  // note that m_slider's tracking state controls how often the valueChanged signal is emitted.
  QObject::connect(&m_slider, &QSlider::valueChanged, [this](int v) {
    this->m_spinner.blockSignals(true);
    this->m_spinner.setValue(v);
    this->m_spinner.blockSignals(false);
  });

  QObject::connect(&m_spinner, QOverload<int>::of(&QSpinBox::valueChanged), [this](int v) {
    // Disabling the spinner only because the valueChanged handler might do a long operation when the spinner is
    // clicked.
    //
    // If the operation is "too long" then the mouserelease of the spinner will kick in after a certain timeout
    // and trigger one more increment.
    //
    // If the owner has disabled tracking, that's a good bet that the increments are
    // expected to be "too long".
    if (!this->m_slider.hasTracking()) {
      this->m_spinner.setEnabled(false);
    }
    this->m_slider.setValue(v);
    if (!this->m_slider.hasTracking()) {
      this->m_spinner.setEnabled(true);
    }
  });

  // only slider will emit the value...
  QObject::connect(&m_slider, SIGNAL(valueChanged(int)), this, SLOT(OnValueChanged(int)));
}

void
QIntSlider::OnValueChanged(int value)
{
  emit valueChanged(value);
}

void
QIntSlider::setSpinnerKeyboardTracking(bool tracking)
{
  m_spinner.setKeyboardTracking(tracking);
}

int
QIntSlider::value(void) const
{
  return m_spinner.value();
}

int
QIntSlider::maximum() const
{
  return m_spinner.maximum();
}

void
QIntSlider::setValue(int value, bool blockSignals)
{
  // only forward the blocksignals flag for one of the two child controls.
  // the other will always block signalling
  m_spinner.blockSignals(true);
  m_slider.blockSignals(blockSignals);
  m_spinner.setValue(value);
  m_slider.setValue(value);
  m_spinner.blockSignals(false);
  m_slider.blockSignals(false);
  if (!blockSignals) {
    emit valueChanged(value);
  }
}

void
QIntSlider::setRange(int rmin, int rmax)
{
  m_slider.setRange(rmin, rmax);
  m_spinner.setRange(rmin, rmax);
}

void
QIntSlider::setSingleStep(int val)
{
  m_spinner.setSingleStep(val);
}

void
QIntSlider::setSuffix(const QString& s)
{
  m_spinner.setSuffix(s);
}

void
QIntSlider::setTickPosition(QSlider::TickPosition position)
{
  m_slider.setTickPosition(position);
}

void
QIntSlider::setTickInterval(int ti)
{
  m_slider.setTickInterval(ti);
}

void
QIntSlider::setTracking(bool enabled)
{
  m_slider.setTracking(enabled);
}
