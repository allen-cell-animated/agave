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
    currentPlane(0), _z(0), _c(0),
    labels(),
    sliders(),
    spinboxes()
{
    QGridLayout *layout = new QGridLayout;


    labels[0] = new QLabel(tr("Plane"));
    labels[1] = new QLabel(tr("Z"));
    labels[2] = new QLabel(tr("mZ"));
    labels[3] = new QLabel(tr("T"));
    labels[4] = new QLabel(tr("mT"));
    labels[5] = new QLabel(tr("C"));
    labels[6] = new QLabel(tr("mC"));

    for (uint16_t i = 0; i < 7; ++i)
    {
        sliders[i] = createSlider();
        spinboxes[i] = createSpinBox();
        layout->addWidget(labels[i], i, 0);
        layout->addWidget(sliders[i], i, 1);
        layout->addWidget(spinboxes[i], i, 2);

        if (i == 0)
        {
            connect(sliders[i], SIGNAL(valueChanged(int)), this, SLOT(sliderChangedPlane(int)));
            connect(spinboxes[i], SIGNAL(valueChanged(int)), this, SLOT(spinBoxChangedPlane(int)));
        }
        else
        {
            connect(sliders[i], SIGNAL(valueChanged(int)), this, SLOT(sliderChangedDimension(int)));
            connect(spinboxes[i], SIGNAL(valueChanged(int)), this, SLOT(spinBoxChangedDimension(int)));
        }
    }

    QWidget *mainWidget = new QWidget(this);
    mainWidget->setLayout(layout);
    setWidget(mainWidget);

    /// Enable widgets.
    setReader(_img, currentPlane);
}

NavigationDock2D::~NavigationDock2D()
{
}

void
NavigationDock2D::setReader(std::shared_ptr<ImageXYZC> img,
                            size_t plane)
{
    this->_img = img;

    if (img)
    {
        // Full dimension sizes.
        uint32_t z = img->sizeZ();
		uint32_t t = 1;
		uint32_t c = img->sizeC();
		// Modulo dimension sizes.
		uint32_t mz = 1;
		uint32_t mt = 1;
		uint32_t mc = 1;
		// Effective dimension sizes.
		uint32_t ez = z / mz;
		uint32_t et = t / mt;
		uint32_t ec = c / mc;

		uint32_t imageCount = z*c*t;
        uint32_t extents[7] = {imageCount, ez, mz, et, mt, ec, mc};
        std::cout << "EXTENTS: " << imageCount << ", " <<  ez << ", " <<  mz << ", " <<  et << ", " <<  mt << ", " <<  ec << ", " <<  mc << "\n";

        for (uint16_t i = 0; i < 7; ++i)
        {
            uint32_t max = extents[i];

            if (max > 1)
            {
                labels[i]->show();
                sliders[i]->show();
                spinboxes[i]->show();
                sliders[i]->setEnabled(true);
                sliders[i]->setRange(0, (int)max - 1);
                spinboxes[i]->setEnabled(true);
                spinboxes[i]->setRange(0, (int)max - 1);
            }
            else
            {
                if (i != 0)
                {
                    labels[i]->hide();
                    sliders[i]->hide();
                    spinboxes[i]->hide();
                }
                sliders[i]->setEnabled(false);
                spinboxes[i]->setEnabled(false);
                sliders[i]->setRange(0, 1);
                spinboxes[i]->setRange(0, 1);
            }
        }
    }
    else
    {
        for (uint16_t i = 1; i < 7; ++i)
        {
            if (i != 0)
            {
                labels[i]->hide();
                sliders[i]->hide();
                spinboxes[i]->hide();
            }
            sliders[i]->setEnabled(false);
            spinboxes[i]->setEnabled(false);
        }
    }

    setPlane(plane);
}

void NavigationDock2D::setZC(size_t z, size_t c) {
	if (z != _z || c != _c) {
		_z = z;
		_c = c;
		emit zcChanged(_z, _c);
	}
}

void
NavigationDock2D::setPlane(size_t plane)
{
    if (plane != currentPlane)
    {
        currentPlane = plane;
        if (_img)
        {
			// Modulo dimension sizes.
			uint32_t mz = 1;
			uint32_t mt = 1;
			uint32_t mc = 1;

			// decompose plane index into a z,c,t triple
			uint32_t z = plane % _img->sizeZ();
			uint32_t c = plane / _img->sizeZ();
			uint32_t t = 0;
			//std::array<size_t, 3> coords(_img->getZCTCoords(plane));

            // Effective and modulo dimension positions
			uint32_t ezv = z / mz;
			uint32_t mzv = z % mz;
			uint32_t etv = t / mt;
			uint32_t mtv = t % mt;
			uint32_t ecv = c / mc;
			uint32_t mcv = c % mc;

			uint32_t values[7] = {plane, ezv, mzv, etv, mtv, ecv, mcv};
            for (uint16_t i = 0; i < 7; ++i)
            {
                sliders[i] -> setValue(static_cast<int>(values[i]));
                spinboxes[i] -> setValue(static_cast<int>(values[i]));
            }
        }
        emit planeChanged(currentPlane);
    }
}

size_t
NavigationDock2D::plane() const
{
    return currentPlane;
}

void
NavigationDock2D::sliderChangedPlane(int plane)
{
    setPlane(static_cast<size_t>(plane));
}

void
NavigationDock2D::spinBoxChangedPlane(int plane)
{
    setPlane(static_cast<size_t>(plane));
}

void
NavigationDock2D::sliderChangedDimension(int /* dim */)
{
	if (_img)
    {
		// Modulo dimension sizes.
		uint32_t mz = 1;
		uint32_t mt = 1;
		uint32_t mc = 1;

        // Current dimension sizes.
		uint32_t ezv = static_cast<uint32_t>(sliders[1]->value());
		uint32_t mzv = static_cast<uint32_t>(sliders[2]->value());
		uint32_t etv = static_cast<uint32_t>(sliders[3]->value());
		uint32_t mtv = static_cast<uint32_t>(sliders[4]->value());
		uint32_t ecv = static_cast<uint32_t>(sliders[5]->value());
		uint32_t mcv = static_cast<uint32_t>(sliders[6]->value());

		uint32_t z = (ezv * mz) + mzv;
		uint32_t t = (etv * mt) + mtv;
		uint32_t c = (ecv * mc) + mcv;

        size_t index = z+c*_img->sizeZ();

		setZC(z, c);
        setPlane(index);
    }
}

void
NavigationDock2D::spinBoxChangedDimension(int /* dim */)
{
    if (_img)
    {
		// Modulo dimension sizes.
		uint32_t mz = 1;
		uint32_t mt = 1;
		uint32_t mc = 1;

        // Current dimension sizes.
		uint32_t ezv = static_cast<uint32_t>(spinboxes[1]->value());
		uint32_t mzv = static_cast<uint32_t>(spinboxes[2]->value());
		uint32_t etv = static_cast<uint32_t>(spinboxes[3]->value());
		uint32_t mtv = static_cast<uint32_t>(spinboxes[4]->value());
		uint32_t ecv = static_cast<uint32_t>(spinboxes[5]->value());
		uint32_t mcv = static_cast<uint32_t>(spinboxes[6]->value());

		uint32_t z = (ezv * mz) + mzv;
		uint32_t t = (etv * mt) + mtv;
		uint32_t c = (ecv * mc) + mcv;

		uint32_t index = z + c*_img->sizeZ();

        setPlane(index);
    }

}

