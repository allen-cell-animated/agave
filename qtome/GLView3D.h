#pragma once

#include <memory>

#include "glm.h"
#include "CameraController.h"
#include "GLWindow.h"
#include "renderlib/CCamera.h"

#include <QElapsedTimer>

class CStatus;
class ImageXYZC;
class QCamera;
class IRenderWindow;
class QTransferFunction;
class Scene;


/**
    * 3D GL view of an image with axes and gridlines.
    */
class GLView3D : public GLWindow
{
    Q_OBJECT

public:

    /**
    * Create a 3D image view.
    *
    * The size and position will be taken from the specified image.
    *
    * @param reader the image reader.
    * @param series the image series.
    * @param parent the parent of this object.
    */
    GLView3D(QCamera* cam,
		QTransferFunction* tran,
		RenderSettings* rs,
		QWidget *parent = 0);

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
	
	void onNewImage(Scene* scene);

	const CCamera& getCamera() { return mCamera; }
public slots:

	void OnUpdateCamera();
	void OnUpdateTransferFunction(void);
	void OnUpdateRenderer(int);

public:

	CStatus* getStatus();

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
	CCamera mCamera;
    CameraController _cameraController;
	QCamera* _camera;
	QTransferFunction* _transferFunction;

    /// Rendering timer.
    QElapsedTimer etimer;

    /// Last mouse position.
    QPoint lastPos;

	RenderSettings* _renderSettings;

	std::unique_ptr<IRenderWindow> _renderer;
	int _rendererType;
};
