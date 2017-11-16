#pragma once

#include <memory>

#include <GLWindow.h>

#include <QtWidgets/QDockWidget>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>

class ImageXYZC;

/**
    * 2D dock widget for plane nagivation.
    *
    * Sliders will be created for each usable dimension, including
    * for Modulo annotations.
    */
class NavigationDock2D : public QDockWidget
{
    Q_OBJECT

public:

    /**
    * Create a 2D navigation view.
    *
    * The size and position will be taken from the specified image.
    *
    * @param parent the parent of this object.
    */
    NavigationDock2D(QWidget *parent = 0);

    /// Destructor.
    ~NavigationDock2D();

    /**
    * Get the current plane for the series.
    *
    * @returns the current plane.
    */
    size_t
    plane() const;

	void
		setReader(std::shared_ptr<ImageXYZC> img,
			size_t plane);

public slots:
    /**
    * Set the current plane for the series.
    *
    * @param plane the image plane.
    */
    void
    setPlane(size_t plane);
	void setZC(size_t z, size_t c);
signals:
    /**
    * Signal change of plane.
    *
    * @param plane the new image plane.
    */
	void
		planeChanged(size_t plane);
	void
		zcChanged(size_t z, size_t c);

private slots:
    /**
    * Update the current plane number (from slider).
    *
    * @param plane the new image plane.
    */
    void
    sliderChangedPlane(int plane);

    /**
    * Update the current plane number (from spinbox).
    *
    * @param plane the new image plane.
    */
    void
    spinBoxChangedPlane(int plane);

    /**
    * Update the current plane number (from dimension slider).
    *
    * @param dim the index of the dimension slider.
    */
    void
    sliderChangedDimension(int dim);

    /**
    * Update the current plane number (from dimension spinbox).
    *
    * @param dim the index of the dimension spinbox.
    */
    void
    spinBoxChangedDimension(int dim);

private:
	std::shared_ptr<ImageXYZC> _img;
	/// The image plane.
    size_t currentPlane;
	size_t _z, _c;
    /// Slider labels [NZTCmZmTmC].
    QLabel *labels[7];
    /// Sliders [NZTCmZmTmC].
    QSlider *sliders[7];
    /// Numeric entries [NZTCmZmTmC].
    QSpinBox *spinboxes[7];
};
