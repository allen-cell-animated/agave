#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_qtome.h"

#include "Camera.h"
#include "GLView3D.h"
#include "NavigationDock2D.h"
#include "TransferFunction.h"
#include "ViewerState.h"

#include "renderlib/RenderSettings.h"
#include "renderlib/AppScene.h"

class QAppearanceDockWidget;
class QCameraDockWidget;
class QStatisticsDockWidget;

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
	void openRecentFile();
	void updateRecentFileActions();
	void quit();
	void view_reset();
	void viewFocusChanged(GLView3D *glView);
	void tabChanged(int index);
	void dumpPythonState();
	void dumpStateToJson();
	void openMeshDialog();
	void openMesh(const QString& file);

private:
	enum { MaxRecentFiles = 8 };

	ViewerState appToViewerState();

	void createActions();
	void createMenus();
	void createToolbars();
	void createDockWindows();
	QDockWidget* createRenderingDock();

	static bool hasRecentFiles();
	void prependToRecentFiles(const QString &fileName);
	void setRecentFilesVisible(bool visible);
	static QString strippedName(const QString &fullFileName);


	QMenu *fileMenu;
	QMenu *viewMenu;

	QToolBar *Cam2DTools;

	QAction *openAction;
	QAction *quitAction;
	QAction *dumpAction;
	QAction *dumpJsonAction;
	QAction *testMeshAction;
	QAction *viewResetAction;

	QSlider *createAngleSlider();
	QSlider *createRangeSlider();
	NavigationDock2D *navigation;

	// THE camera parameter container
	QCamera _camera;
	// Camera UI
	QCameraDockWidget* cameradock;
	
	QTransferFunction _transferFunction;
	QAppearanceDockWidget* appearanceDockWidget;

	QStatisticsDockWidget* statisticsDockWidget;

	QTabWidget *tabs;
	GLView3D *glView;

	QMetaObject::Connection navigationChanged;
	QMetaObject::Connection navigationZCChanged;
	QMetaObject::Connection navigationUpdate;

	// THE underlying render settings container.
	// There is only one of these.  The app owns it and hands refs to the ui widgets and the renderer.
	// if renderer is on a separate thread, then this will need a mutex guard
	// any direct programmatic changes to this obj need to be pushed to the UI as well.
	RenderSettings _renderSettings;

	// the app owns a scene.
	// scene gets sent down to the renderer.
	Scene _appScene;

	QAction *recentFileActs[MaxRecentFiles];
	QAction *recentFileSeparator;
	QAction *recentFileSubMenuAct;

	QString _currentFilePath;
};
