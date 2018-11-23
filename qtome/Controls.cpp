#include "Controls.h"
#include "Stable.h" // for GetIcon

#include <QtGlobal>
#include <QtGui/QPainter>
#include <QtGui/QPaintEvent>
#include <QtWidgets/QColorDialog>

QColorPushButton::QColorPushButton(QWidget* pParent) :
	QPushButton(pParent),
	m_Margin(5),
	m_Radius(4),
	m_Color(Qt::gray)
{
	setText("");
}

QSize QColorPushButton::sizeHint() const
{
	return QSize(20, 20);
}

void QColorPushButton::paintEvent(QPaintEvent* pPaintEvent)
{
	setText("");

	QPushButton::paintEvent(pPaintEvent);

	QPainter Painter(this);

	// Get button rectangle
	QRect ColorRectangle = pPaintEvent->rect();

	// Deflate it
	ColorRectangle.adjust(m_Margin, m_Margin, -m_Margin, -m_Margin);

	// Use anti aliasing
	Painter.setRenderHint(QPainter::Antialiasing);

	// Rectangle styling
	Painter.setBrush(QBrush(isEnabled() ? m_Color : Qt::lightGray));
	Painter.setPen(QPen(isEnabled() ? QColor(25, 25, 25) : Qt::darkGray, 0.5));

	// Draw
	Painter.drawRoundedRect(ColorRectangle, m_Radius, Qt::AbsoluteSize);
}

void QColorPushButton::mousePressEvent(QMouseEvent* pEvent)
{
	QColorDialog ColorDialog;

	connect(&ColorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));

	ColorDialog.setWindowIcon(GetIcon("color--pencil"));
	ColorDialog.setCurrentColor(m_Color);
	ColorDialog.exec();

	disconnect(&ColorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));
}

int QColorPushButton::GetMargin(void) const
{
	return m_Margin;
}

void QColorPushButton::SetMargin(const int& Margin)
{
	m_Margin = m_Margin;
	update();
}

int QColorPushButton::GetRadius(void) const
{
	return m_Radius;
}

void QColorPushButton::SetRadius(const int& Radius)
{
	m_Radius = m_Radius;
	update();
}

QColor QColorPushButton::GetColor(void) const
{
	return m_Color;
}

void QColorPushButton::SetColor(const QColor& Color, bool BlockSignals)
{
	blockSignals(BlockSignals);

	m_Color = Color;
	update();

	blockSignals(false);
}

void QColorPushButton::OnCurrentColorChanged(const QColor& Color)
{
	SetColor(Color);

	emit currentColorChanged(m_Color);
}

QColorSelector::QColorSelector(QWidget* pParent /*= NULL*/) :
	QFrame(pParent),
	m_ColorButton(),
	m_ColorCombo()
{
	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(&m_ColorButton, 0, 0, Qt::AlignLeft);
//	m_MainLayout.addWidget(&m_ColorCombo, 0, 1);

	m_MainLayout.setContentsMargins(0, 0, 0, 0);
	
	m_ColorButton.setFixedWidth(30);

	QObject::connect(&m_ColorButton, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));
}

QColor QColorSelector::GetColor(void) const
{
	return m_ColorButton.GetColor();
}

void QColorSelector::SetColor(const QColor& Color, bool BlockSignals /*= false*/)
{
	m_ColorButton.SetColor(Color, BlockSignals);
}

void QColorSelector::OnCurrentColorChanged(const QColor& Color)
{
	emit currentColorChanged(Color);
}

QDoubleSlider::QDoubleSlider(QWidget* pParent /*= NULL*/) :
	QSlider(pParent),
	m_Multiplier(10000.0)
{
	connect(this, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));

	setSingleStep(1);

	setOrientation(Qt::Horizontal);
	setFocusPolicy(Qt::NoFocus);
}

void QDoubleSlider::setValue(int Value)
{
	emit valueChanged((double)Value / m_Multiplier);
}

void QDoubleSlider::setValue(double Value, bool BlockSignals)
{
	QSlider::blockSignals(BlockSignals);

	QSlider::setValue(Value * m_Multiplier);

 	if (!BlockSignals)
 		emit valueChanged(Value);

	QSlider::blockSignals(false);
}

void QDoubleSlider::setRange(double Min, double Max)
{
	QSlider::setRange(Min * m_Multiplier, Max * m_Multiplier);

	emit rangeChanged(Min, Max);
}

void QDoubleSlider::setMinimum(double Min)
{
	QSlider::setMinimum(Min * m_Multiplier);

	emit rangeChanged(minimum(), maximum());
}

double QDoubleSlider::minimum() const
{
	return QSlider::minimum() / m_Multiplier;
}

void QDoubleSlider::setMaximum(double Max)
{
	QSlider::setMaximum(Max * m_Multiplier);

	emit rangeChanged(minimum(), maximum());
}

double QDoubleSlider::maximum() const
{
	return QSlider::maximum() / m_Multiplier;
}

double QDoubleSlider::value() const
{
	int Value = QSlider::value();
	return (double)Value / m_Multiplier;
}

QSize QDoubleSpinner::sizeHint() const
{
	return QSize(90, 20);
}

QDoubleSpinner::QDoubleSpinner(QWidget* pParent /*= NULL*/) :
	QDoubleSpinBox(pParent)
{
}

void QDoubleSpinner::setValue(double Value, bool BlockSignals)
{
	blockSignals(BlockSignals);

	QDoubleSpinBox::setValue(Value);

	blockSignals(false);
}

QInputDialogEx::QInputDialogEx(QWidget* pParent /*= NULL*/, Qt::WindowFlags Flags /*= 0*/) :
	QInputDialog(pParent, Flags)
{
	setWindowIcon(GetIcon("pencil-field"));
}

QSize QInputDialogEx::sizeHint() const
{
	return QSize(350, 60);
}

QNumericSlider::QNumericSlider(QWidget* pParent /*= NULL*/) :
	QWidget(pParent),
	m_slider(),
	m_spinner()
{
	setLayout(&m_layout);

	m_slider.setOrientation(Qt::Horizontal);
	m_spinner.setDecimals(4);

	// entire control is one single row.
	// slider is 3/4, spinner is 1/4 of the width
	const int sliderratio = 4;
	m_layout.addWidget(&m_slider, 0, 0, 1, sliderratio-1);
	m_layout.addWidget(&m_spinner, 0, sliderratio-1, 1, 1);

	m_layout.setContentsMargins(0, 0, 0, 0);

	// keep slider and spinner in sync
	QObject::connect(&m_slider, QOverload<double>::of(&QDoubleSlider::valueChanged),
		[this](double v) { this->m_spinner.setValue(v, true);  });
	QObject::connect(&m_spinner, QOverload<double>::of(&QDoubleSpinner::valueChanged),
		[this](double v) { this->m_slider.setValue(v);  });

	// only slider will update the value...
	QObject::connect(&m_slider, SIGNAL(valueChanged(double)), this, SLOT(OnValueChanged(double)));
}

void QNumericSlider::OnValueChanged(double value) {
	emit valueChanged(value);
}

double QNumericSlider::value(void) const {
	return m_spinner.value();
}

void QNumericSlider::setValue(double value, bool BlockSignals)
{
	// only forward the blocksignals flag for one of the two child controls.
	// the other will always block signalling
	m_spinner.setValue(value, true);
	m_slider.setValue(value, BlockSignals);
}

void QNumericSlider::setRange(double rmin, double rmax)
{
	m_slider.setRange(rmin, rmax);
	m_spinner.setRange(rmin, rmax);
}

void QNumericSlider::setDecimals(int decimals)
{
	m_spinner.setDecimals(decimals);
}

void QNumericSlider::setSuffix(const QString& s) {
	m_spinner.setSuffix(s);
}
