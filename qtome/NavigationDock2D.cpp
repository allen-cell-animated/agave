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
    m_c(0),
    m_label(),
    m_slider(),
    m_spinbox()
{
    QGridLayout *layout = new QGridLayout;

    m_label = new QLabel(tr("C"));
	m_slider = createSlider();
	m_spinbox = createSpinBox();
	layout->addWidget(m_label, 0, 0);
	layout->addWidget(m_slider, 0, 1);
	layout->addWidget(m_spinbox, 0, 2);
    connect(m_slider, SIGNAL(valueChanged(int)), this, SLOT(sliderChangedDimension(int)));
    connect(m_spinbox, SIGNAL(valueChanged(int)), this, SLOT(spinBoxChangedDimension(int)));

    QWidget *mainWidget = new QWidget(this);
    mainWidget->setLayout(layout);
    setWidget(mainWidget);

    /// Enable widgets.
    setReader(m_img);
}

NavigationDock2D::~NavigationDock2D()
{
}

void
NavigationDock2D::setReader(std::shared_ptr<ImageXYZC> img)
{
    this->m_img = img;

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
            m_label->show();
            m_slider->show();
            m_spinbox->show();
            m_slider->setEnabled(true);
            m_slider->setRange(0, (int)max - 1);
            m_spinbox->setEnabled(true);
            m_spinbox->setRange(0, (int)max - 1);
        }
        else
        {
            m_label->hide();
			m_slider->hide();
			m_spinbox->hide();
	        m_slider->setEnabled(false);
		    m_spinbox->setEnabled(false);
            m_slider->setRange(0, 1);
            m_spinbox->setRange(0, 1);
        }
    }
    else
    {
        m_label->hide();
        m_slider->hide();
        m_spinbox->hide();
        m_slider->setEnabled(false);
        m_spinbox->setEnabled(false);
    }
}

void NavigationDock2D::setC(size_t c) {
	if (c != m_c) {
		m_c = c;
		emit cChanged(m_c);
	}
}

void
NavigationDock2D::sliderChangedDimension(int c)
{
	if (m_img)
    {
		m_spinbox->setValue(c);
		setC(c);
    }
}

void
NavigationDock2D::spinBoxChangedDimension(int c)
{
    if (m_img)
    {
		m_slider->setValue(c);
		setC(c);
	}

}

