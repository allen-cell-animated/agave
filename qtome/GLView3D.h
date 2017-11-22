#pragma once

#include <memory>

#include "glm.h"
#include "CameraController.h"
#include "GLWindow.h"

#include <QElapsedTimer>

#include "renderlib/Camera.h"

class ImageXYZC;
class QCamera;
class IRenderWindow;
class QTransferFunction;



/**
    * 3D GL view of an image with axes and gridlines.
    */
class GLView3D : public GLWindow
{
    Q_OBJECT

public:
    /// Mouse behaviour.
    enum MouseMode
    {
        MODE_ZOOM,  ///< Zoom in and out.
        MODE_PAN,   ///< Pan in x and y.
        MODE_ROTATE ///< Rotate around point in z.
    };

    /**
    * Create a 2D image view.
    *
    * The size and position will be taken from the specified image.
    *
    * @param reader the image reader.
    * @param series the image series.
    * @param parent the parent of this object.
    */
    GLView3D(std::shared_ptr<ImageXYZC>  img,
		QCamera* cam,
		QTransferFunction* tran,
		CScene* scene,
		QWidget                                                *parent = 0);

    /// Destructor.
    ~GLView3D();

    /**
    * Get window minimum size hint.
    *
    * @returns the size hint.
    */
    QSize minimumSizeHint() const;

    /**
    * Get window size hint.
    *
    * @returns the size hint.
    */
    QSize sizeHint() const;

public slots:
    /**
    * Set zoom factor.
    *
    * @param zoom the zoom factor (pixel drag distance).
    */
    void
    setZoom(int zoom);

    /**
    * Set x translation factor.
    *
    * @param xtran x translation factor (pixels).
    */
    void
    setXTranslation(int xtran);

    /**
    * Set y translation factor.
    *
    * @param ytran y translation factor (pixels).
    */
    void
    setYTranslation(int ytran);

    /**
    * Set z rotation factor.
    *
    * @param angle z rotation factor (pixel drag distance).
    */
    void
    setZRotation(int angle);

	void
	setC(size_t c);

    /**
    * Set mouse behaviour mode.
    *
    * @param mode the behaviour mode to set.
    */
    void
    setMouseMode(MouseMode mode);
	void OnUpdateCamera();
	void OnUpdateTransferFunction(void);
	void OnUpdateRenderer(int);

public:

    /**
    * Get zoom factor.
    *
    * @returns the zoom factor.
    */
    int
    getZoom() const;

    /**
    * Get x translation factor.
    *
    * @returns the x translation factor.
    */
    int
    getXTranslation() const;

    /**
    * Get y translation factor.
    *
    * @returns the y translation factor.
    */
    int
    getYTranslation() const;

    /**
    * Get z rotation factor.
    *
    * @returns the z rotation factor.
    */
    int
    getZRotation() const;


	size_t getC() const;


    /**
    * Get mouse behaviour mode.
    *
    * @returns the behaviour mode.
    */
    MouseMode
    getMouseMode() const;


	std::shared_ptr<ImageXYZC> getImage() { return _img; }

signals:
    /**
    * Signal zoom level changed.
    *
    * @param zoom the new zoom level.
    */
    void
    zoomChanged(int zoom);

    /**
    * Signal x translation changed.
    *
    * @param xtran the new x translation.
    */
    void
    xTranslationChanged(int xtran);

    /**
    * Signal y translation changed.
    *
    * @param ytran the new y translation.
    */
    void
    yTranslationChanged(int ytran);

    /**
    * Signal z rotation changed.
    *
    * @param angle the new z rotation.
    */
    void
    zRotationChanged(int angle);


protected:
    /// Set up GL context and subsidiary objects.
    void
    initialize();

    using GLWindow::render;

    /// Render the scene with the current view settings.
    void
    render();

    /// Resize the view.
    void
    resize();

    /**
    * Handle mouse button press events.
    *
    * Action depends upon the mouse behaviour mode.
    *
    * @param event the event to handle.
    */
    void
    mousePressEvent(QMouseEvent *event);
    void
    mouseReleaseEvent(QMouseEvent *event);

    /**
    * Handle mouse button movement events.
    *
    * Action depends upon the mouse behaviour mode.
    *
    * @param event the event to handle.
    */
    void
    mouseMoveEvent(QMouseEvent *event);

    /**
    * Handle timer events.
    *
    * Used to update scene properties and trigger a render pass.
    *
    * @param event the event to handle.
    */
    void
    timerEvent (QTimerEvent *event);

private:
    CameraController _cameraController;
	QCamera* _camera;
	QTransferFunction* _transferFunction;

    /// Current projection
    Camera camera;
    /// Current mouse behaviour.
    MouseMode mouseMode;
    /// Rendering timer.
    QElapsedTimer etimer;
    /// Current plane.
	size_t _c;

    /// Last mouse position.
    QPoint lastPos;

	std::shared_ptr<ImageXYZC> _img;
	CScene* _scene;

	std::unique_ptr<IRenderWindow> _renderer;
};
