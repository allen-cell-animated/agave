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

	void
		setReader(std::shared_ptr<ImageXYZC> img);

public slots:
	void sliderChangedDimension(int);
	void spinBoxChangedDimension(int);
    /**
    * Set the current plane for the series.
    *
    * @param plane the image plane.
    */
	void setC(size_t c);
signals:
	void
		cChanged(size_t c);

private:
	std::shared_ptr<ImageXYZC> m_img;
	/// The image plane.
	size_t m_c;
    /// Slider labels [NZTCmZmTmC].
    QLabel *m_label;
    /// Sliders [NZTCmZmTmC].
    QSlider *m_slider;
    /// Numeric entries [NZTCmZmTmC].
    QSpinBox *m_spinbox;
};
