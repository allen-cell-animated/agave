#include <QtGui/QMouseEvent>

#include "ImageXYZC.h"

#include <NavigationDock2D.h>

#include <array>
#include <cmath>
#include <iostream>

#include <QtWidgets/QGridLayout>

namespace
{

  QSlider *
  createSlider()
  {
    QSlider *slider = new QSlider(Qt::Horizontal);
    slider->setRange(0, 1);
    slider->setSingleStep(1);
    slider->setPageStep(8);
    slider->setTickInterval(8);
    slider->setTickPosition(QSlider::TicksRight);
    return slider;
  }

  QSpinBox *
  createSpinBox()
  {
    QSpinBox *slider = new QSpinBox;
    slider->setRange(0, 1);
    slider->setSingleStep(1);
    slider->setButtonSymbols(QAbstractSpinBox::NoButtons);
    return slider;
  }
}


NavigationDock2D::NavigationDock2D(QWidget *parent):
    QDockWidget(tr("Navigation"), parent),
    _c(0),
    label(),
    slider(),
    spinbox()
{
    QGridLayout *layout = new QGridLayout;

    label = new QLabel(tr("C"));
	slider = createSlider();
	spinbox = createSpinBox();
	layout->addWidget(label, 0, 0);
	layout->addWidget(slider, 0, 1);
	layout->addWidget(spinbox, 0, 2);
    connect(slider, SIGNAL(valueChanged(int)), this, SLOT(sliderChangedDimension(int)));
    connect(spinbox, SIGNAL(valueChanged(int)), this, SLOT(spinBoxChangedDimension(int)));

    QWidget *mainWidget = new QWidget(this);
    mainWidget->setLayout(layout);
    setWidget(mainWidget);

    /// Enable widgets.
    setReader(_img);
}

NavigationDock2D::~NavigationDock2D()
{
}

void
NavigationDock2D::setReader(std::shared_ptr<ImageXYZC> img)
{
    this->_img = img;

    if (img)
    {
        // Full dimension sizes.
		uint32_t c = img->sizeC();
		// Modulo dimension sizes.
		uint32_t mc = 1;
		// Effective dimension sizes.
		uint32_t ec = c / mc;

        uint32_t extent = ec;

        uint32_t max = extent;

        if (max > 1)
        {
            label->show();
            slider->show();
            spinbox->show();
            slider->setEnabled(true);
            slider->setRange(0, (int)max - 1);
            spinbox->setEnabled(true);
            spinbox->setRange(0, (int)max - 1);
        }
        else
        {
            label->hide();
			slider->hide();
			spinbox->hide();
	        slider->setEnabled(false);
		    spinbox->setEnabled(false);
            slider->setRange(0, 1);
            spinbox->setRange(0, 1);
        }
    }
    else
    {
        label->hide();
        slider->hide();
        spinbox->hide();
        slider->setEnabled(false);
        spinbox->setEnabled(false);
    }
}

void NavigationDock2D::setC(size_t c) {
	if (c != _c) {
		_c = c;
		emit cChanged(_c);
	}
}

void
NavigationDock2D::sliderChangedDimension(int c)
{
	if (_img)
    {
		spinbox->setValue(c);
		setC(c);
    }
}

void
NavigationDock2D::spinBoxChangedDimension(int c)
{
    if (_img)
    {
		slider->setValue(c);
		setC(c);
	}

}

