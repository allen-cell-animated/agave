#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_qtome.h"

#include "Camera.h"
#include "GLView3D.h"
#include "NavigationDock2D.h"
#include "TransferFunction.h"

#include "Scene.h"

class QAppearanceDockWidget;
class QCameraDockWidget;

class qtome : public QMainWindow
{
	Q_OBJECT

public:
	qtome(QWidget *parent = Q_NULLPTR);

private:
	Ui::qtomeClass ui;

private slots:
	void open();
	void open(const QString& file);
	void quit();
	void view_reset();
	void view_zoom();
	void view_pan();
	void view_rotate();
	void viewFocusChanged(GLView3D *glView);
	void tabChanged(int index);

private:
	void createActions();
	void createMenus();
	void createToolbars();
	void createDockWindows();

	QMenu *fileMenu;
	QMenu *viewMenu;

	QToolBar *Cam2DTools;

	QAction *openAction;
	QAction *quitAction;

	QAction *viewResetAction;

	QActionGroup *viewActionGroup;
	QAction *viewZoomAction;
	QAction *viewPanAction;
	QAction *viewRotateAction;

	QSlider *createAngleSlider();
	QSlider *createRangeSlider();
	NavigationDock2D *navigation;

	// THE camera parameter container
	QCamera _camera;
	// Camera UI
	QCameraDockWidget* cameradock;
	
	QTransferFunction _transferFunction;
	QAppearanceDockWidget* appearanceDockWidget;

	QTabWidget *tabs;
	GLView3D *glView;
	QSlider *minSlider;
	QSlider *maxSlider;

	QMetaObject::Connection minSliderChanged;
	QMetaObject::Connection minSliderUpdate;
	QMetaObject::Connection maxSliderChanged;
	QMetaObject::Connection maxSliderUpdate;
	QMetaObject::Connection navigationChanged;
	QMetaObject::Connection navigationZCChanged;
	QMetaObject::Connection navigationUpdate;

	// THE underlying render settings container.
	// There is only one of these.  The app owns it and hands refs to the ui widgets and the renderer.
	// if renderer is on a separate thread, then this will need a mutex guard
	// any direct programmatic changes to this obj need to be pushed to the UI as well.
	CScene _renderSettings;

};
