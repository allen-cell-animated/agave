#include <QtGui/QMouseEvent>

#include <array>
#include <cmath>

#include <NavigationDock2D.h>

#include <iostream>

#include <QtWidgets/QGridLayout>

using ome::files::dimension_size_type;

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
    currentPlane(), _z(0), _c(0),
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
    setReader(reader, series);
}

NavigationDock2D::~NavigationDock2D()
{
}

void
NavigationDock2D::setReader(std::shared_ptr<ome::files::FormatReader> reader,
                            ome::files::dimension_size_type                   series,
                            ome::files::dimension_size_type                   plane)
{
    this->reader = reader;
    this->series = series;

    if (reader)
    {
        ome::files::dimension_size_type oldseries = reader->getSeries();
        reader->setSeries(series);
        dimension_size_type imageCount = reader->getImageCount();
        // Full dimension sizes.
        dimension_size_type z = reader->getSizeZ();
        dimension_size_type t = reader->getSizeT();
        dimension_size_type c = reader->getSizeC();
        // Modulo dimension sizes.
        dimension_size_type mz = reader->getModuloZ().size();
        dimension_size_type mt = reader->getModuloT().size();
        dimension_size_type mc = reader->getModuloC().size();
        // Effective dimension sizes.
        dimension_size_type ez = z / mz;
        dimension_size_type et = t / mt;
        dimension_size_type ec = c / mc;
        reader->setSeries(oldseries);

        dimension_size_type extents[7] = {imageCount, ez, mz, et, mt, ec, mc};
        std::cout << "EXTENTS: " << imageCount << ", " <<  ez << ", " <<  mz << ", " <<  et << ", " <<  mt << ", " <<  ec << ", " <<  mc << "\n";

        for (uint16_t i = 0; i < 7; ++i)
        {
            dimension_size_type max = extents[i];

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
NavigationDock2D::setPlane(ome::files::dimension_size_type plane)
{
    if (plane != currentPlane)
    {
        currentPlane = plane;

        if (reader)
        {
            ome::files::dimension_size_type oldseries = reader->getSeries();
            reader->setSeries(series);
            // Modulo dimension sizes.
            dimension_size_type mz = reader->getModuloZ().size();
            dimension_size_type mt = reader->getModuloT().size();
            dimension_size_type mc = reader->getModuloC().size();

            std::array<dimension_size_type, 3> coords(reader->getZCTCoords(plane));
            reader->setSeries(oldseries);

            // Effective and modulo dimension positions
            dimension_size_type ezv = coords[0] / mz;
            dimension_size_type mzv = coords[0] % mz;
            dimension_size_type etv = coords[2] / mt;
            dimension_size_type mtv = coords[2] % mt;
            dimension_size_type ecv = coords[1] / mc;
            dimension_size_type mcv = coords[1] % mc;

            dimension_size_type values[7] = {plane, ezv, mzv, etv, mtv, ecv, mcv};
            for (uint16_t i = 0; i < 7; ++i)
            {
                sliders[i] -> setValue(static_cast<int>(values[i]));
                spinboxes[i] -> setValue(static_cast<int>(values[i]));
            }
        }

        emit planeChanged(currentPlane);
    }
}

ome::files::dimension_size_type
NavigationDock2D::plane() const
{
    return currentPlane;
}

void
NavigationDock2D::sliderChangedPlane(int plane)
{
    setPlane(static_cast<dimension_size_type>(plane));
}

void
NavigationDock2D::spinBoxChangedPlane(int plane)
{
    setPlane(static_cast<dimension_size_type>(plane));
}

void
NavigationDock2D::sliderChangedDimension(int /* dim */)
{
    if (reader)
    {
        ome::files::dimension_size_type oldseries = reader->getSeries();
        reader->setSeries(series);
        // Modulo dimension sizes.
        dimension_size_type mz = reader->getModuloZ().size();
        dimension_size_type mt = reader->getModuloT().size();
        dimension_size_type mc = reader->getModuloC().size();

        // Current dimension sizes.
        dimension_size_type ezv = static_cast<dimension_size_type>(sliders[1]->value());
        dimension_size_type mzv = static_cast<dimension_size_type>(sliders[2]->value());
        dimension_size_type etv = static_cast<dimension_size_type>(sliders[3]->value());
        dimension_size_type mtv = static_cast<dimension_size_type>(sliders[4]->value());
        dimension_size_type ecv = static_cast<dimension_size_type>(sliders[5]->value());
        dimension_size_type mcv = static_cast<dimension_size_type>(sliders[6]->value());

        dimension_size_type z = (ezv * mz) + mzv;
        dimension_size_type t = (etv * mt) + mtv;
        dimension_size_type c = (ecv * mc) + mcv;

        dimension_size_type index = reader->getIndex(z, c, t);

        reader->setSeries(oldseries);

		setZC(z, c);
        setPlane(index);
    }
}

void
NavigationDock2D::spinBoxChangedDimension(int /* dim */)
{
    if (reader)
    {
        ome::files::dimension_size_type oldseries = reader->getSeries();
        reader->setSeries(series);
        // Modulo dimension sizes.
        dimension_size_type mz = reader->getModuloZ().size();
        dimension_size_type mt = reader->getModuloT().size();
        dimension_size_type mc = reader->getModuloC().size();

        // Current dimension sizes.
        dimension_size_type ezv = static_cast<dimension_size_type>(spinboxes[1]->value());
        dimension_size_type mzv = static_cast<dimension_size_type>(spinboxes[2]->value());
        dimension_size_type etv = static_cast<dimension_size_type>(spinboxes[3]->value());
        dimension_size_type mtv = static_cast<dimension_size_type>(spinboxes[4]->value());
        dimension_size_type ecv = static_cast<dimension_size_type>(spinboxes[5]->value());
        dimension_size_type mcv = static_cast<dimension_size_type>(spinboxes[6]->value());

        dimension_size_type z = (ezv * mz) + mzv;
        dimension_size_type t = (etv * mt) + mtv;
        dimension_size_type c = (ecv * mc) + mcv;

        dimension_size_type index = reader->getIndex(z, c, t);

        reader->setSeries(oldseries);

        setPlane(index);
    }
}

