// Include first to avoid clash with Windows headers pulled in via
// QtCore/qt_windows.h; they define VOID and HALFTONE which clash with
// the TIFF enums.
#include <memory>

#include "qtome.h"

#include "renderlib/FileReader.h"
#include "renderlib/Logging.h"

#include "AppearanceDockWidget.h"
#include "CameraDockWidget.h"
#include "GLContainer.h"

#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QAction>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QFileDialog>

#include <boost/filesystem/path.hpp>

qtome::qtome(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	createActions();
	createMenus();
	createToolbars();
	createDockWindows();
	setDockOptions(AllowTabbedDocks);

	tabs = new QTabWidget(this);

	QHBoxLayout *mainLayout = new QHBoxLayout;
	mainLayout->addWidget(tabs);

	QWidget *central = new QWidget(this);
	central->setLayout(mainLayout);

	setCentralWidget(central);

	connect(tabs, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));


	setWindowTitle(tr("OME-Files GLView"));

	open("C:\\Users\\danielt.ALLENINST\\Downloads\\AICS-12_269_4.ome.tif");
	//open("/home/danielt/Downloads/AICS-12_269_4.ome.tif");

}

void qtome::createActions()
{
	boost::filesystem::path iconpath(QCoreApplication::applicationDirPath().toStdString());

	openAction = new QAction(tr("&Open image..."), this);
	openAction->setShortcuts(QKeySequence::Open);
	openAction->setStatusTip(tr("Open an existing image file"));
	connect(openAction, SIGNAL(triggered()), this, SLOT(open()));

	quitAction = new QAction(tr("&Quit"), this);
	quitAction->setShortcuts(QKeySequence::Quit);
	quitAction->setStatusTip(tr("Quit the application"));
	connect(quitAction, SIGNAL(triggered()), this, SLOT(quit()));

	viewResetAction = new QAction(tr("&Reset"), this);
	viewResetAction->setShortcut(QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_R));
	viewResetAction->setStatusTip(tr("Reset the current view"));
	QIcon reset_icon(QString((iconpath / "actions/ome-reset2d.svg").string().c_str()));
	viewResetAction->setIcon(reset_icon);
	viewResetAction->setEnabled(false);
	connect(viewResetAction, SIGNAL(triggered()), this, SLOT(view_reset()));

	viewZoomAction = new QAction(tr("&Zoom"), this);
	viewZoomAction->setCheckable(true);
	viewZoomAction->setShortcut(QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_Z));
	viewZoomAction->setStatusTip(tr("Zoom the current view"));
	QIcon zoom_icon(QString((iconpath / "actions/ome-zoom2d.svg").string().c_str()));
	viewZoomAction->setIcon(zoom_icon);
	viewZoomAction->setEnabled(false);
	connect(viewZoomAction, SIGNAL(triggered()), this, SLOT(view_zoom()));

	viewPanAction = new QAction(tr("&Pan"), this);
	viewPanAction->setCheckable(true);
	viewPanAction->setShortcut(QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_P));
	viewPanAction->setStatusTip(tr("Pan the current view"));
	QIcon pan_icon(QString((iconpath / "actions/ome-pan2d.svg").string().c_str()));
	viewPanAction->setIcon(pan_icon);
	viewPanAction->setEnabled(false);
	connect(viewPanAction, SIGNAL(triggered()), this, SLOT(view_pan()));

	viewRotateAction = new QAction(tr("Rota&te"), this);
	viewRotateAction->setCheckable(true);
	viewRotateAction->setShortcut(QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_T));
	viewRotateAction->setStatusTip(tr("Rotate the current view"));
	QIcon rotate_icon(QString((iconpath / "actions/ome-rotate2d.svg").string().c_str()));
	viewRotateAction->setIcon(rotate_icon);
	viewRotateAction->setEnabled(false);
	connect(viewRotateAction, SIGNAL(triggered()), this, SLOT(view_rotate()));

	viewActionGroup = new QActionGroup(this);
	viewActionGroup->addAction(viewZoomAction);
	viewActionGroup->addAction(viewPanAction);
	viewActionGroup->addAction(viewRotateAction);
	viewRotateAction->setChecked(true);
}

void qtome::createMenus()
{
	fileMenu = menuBar()->addMenu(tr("&File"));
	fileMenu->addAction(openAction);
	fileMenu->addSeparator();
	fileMenu->addAction(quitAction);

	viewMenu = menuBar()->addMenu(tr("&View"));
	viewMenu->addAction(viewResetAction);
	fileMenu->addSeparator();
	viewMenu->addAction(viewZoomAction);
	viewMenu->addAction(viewPanAction);
	viewMenu->addAction(viewRotateAction);
}

void qtome::createToolbars()
{
	Cam2DTools = new QToolBar("2D Camera", this);
	addToolBar(Qt::TopToolBarArea, Cam2DTools);
	Cam2DTools->addAction(viewResetAction);
	Cam2DTools->addAction(viewZoomAction);
	Cam2DTools->addAction(viewPanAction);
	Cam2DTools->addAction(viewRotateAction);

	viewMenu->addSeparator();
	viewMenu->addAction(Cam2DTools->toggleViewAction());
}

QDockWidget* qtome::createRenderingDock() {
	QDockWidget *dock = new QDockWidget(tr("Rendering"), this);
	dock->setAllowedAreas(Qt::AllDockWidgetAreas);

	QGridLayout *layout = new QGridLayout;

	QLabel *minLabel = new QLabel(tr("Min"));
	QLabel *maxLabel = new QLabel(tr("Max"));
	//minSlider = createRangeSlider();
	//maxSlider = createRangeSlider();

	layout->addWidget(minLabel, 0, 0);
	//layout->addWidget(minSlider, 0, 1);
	layout->addWidget(maxLabel, 1, 0);
	//layout->addWidget(maxSlider, 1, 1);

	QWidget *mainWidget = new QWidget(this);
	mainWidget->setLayout(layout);
	dock->setWidget(mainWidget);
	return dock;
}

void qtome::createDockWindows()
{
	navigation = new NavigationDock2D(this);
	navigation->setAllowedAreas(Qt::AllDockWidgetAreas);
	addDockWidget(Qt::BottomDockWidgetArea, navigation);

	cameradock = new QCameraDockWidget(this, &_camera, &_renderSettings);
	cameradock->setAllowedAreas(Qt::AllDockWidgetAreas);
	addDockWidget(Qt::RightDockWidgetArea, cameradock);

	appearanceDockWidget = new QAppearanceDockWidget(this, &_transferFunction, &_renderSettings);
	appearanceDockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);
	addDockWidget(Qt::RightDockWidgetArea, appearanceDockWidget);

	viewMenu->addSeparator();
	viewMenu->addAction(navigation->toggleViewAction());

//	QDockWidget* dock = createRenderingDock();
//	addDockWidget(Qt::BottomDockWidgetArea, dock);
//	viewMenu->addAction(dock->toggleViewAction());
}

QSlider *qtome::createAngleSlider()
{
	QSlider *slider = new QSlider(Qt::Vertical);
	slider->setRange(0, 365 * 16);
	slider->setSingleStep(16);
	slider->setPageStep(8 * 16);
	slider->setTickInterval(8 * 16);
	slider->setTickPosition(QSlider::TicksRight);
	return slider;
}

QSlider *qtome::createRangeSlider()
{
	QSlider *slider = new QSlider(Qt::Horizontal);
	slider->setRange(0, 255 * 16);
	slider->setSingleStep(16);
	slider->setPageStep(8 * 16);
	slider->setTickInterval(8 * 16);
	slider->setTickPosition(QSlider::TicksRight);
	return slider;
}

void qtome::open()
{
	QString file = QFileDialog::getOpenFileName(this,
		tr("Open Image"),
		QString(),
		QString(),
		0,
		QFileDialog::DontResolveSymlinks);

	if (!file.isEmpty())
		open(file);
}

void qtome::open(const QString& file)
{
	QFileInfo info(file);
	if (info.exists())
	{

		FileReader fileReader(QCoreApplication::applicationDirPath().toStdString() + "/ome");
//		std::shared_ptr<ome::files::FormatReader> reader = fileReader.open(file.toStdString());


		//std::shared_ptr<ImageXYZC> image = fileReader.openToImage(file.toStdString());
		std::shared_ptr<ImageXYZC> image = fileReader.loadOMETiff_4D(file.toStdString());

		GLView3D *newGlView = new GLView3D(image, &_camera, &_transferFunction, &_renderSettings, this);
		QWidget *glContainer = new GLContainer(this, newGlView);
		newGlView->setObjectName("glcontainer");
		// We need a minimum size or else the size defaults to zero.
		glContainer->setMinimumSize(512, 512);
		tabs->addTab(glContainer, info.fileName());
		newGlView->setC(0);
	}
}

void qtome::viewFocusChanged(GLView3D *newGlView)
{
	if (glView == newGlView)
		return;

	disconnect(navigationChanged);
	disconnect(navigationZCChanged);
	disconnect(navigationUpdate);

	viewResetAction->setEnabled(false);
	viewZoomAction->setEnabled(false);
	viewPanAction->setEnabled(false);
	viewRotateAction->setEnabled(false);

	if (newGlView)
	{
		navigation->setReader(newGlView->getImage());
		navigationZCChanged = connect(navigation, SIGNAL(cChanged(size_t)), newGlView, SLOT(setC(size_t)));

	}
	else
	{
		navigation->setReader(std::shared_ptr<ImageXYZC>());
	}

	bool enable(newGlView != 0);

	viewResetAction->setEnabled(enable);
	viewZoomAction->setEnabled(enable);
	viewPanAction->setEnabled(enable);
	viewRotateAction->setEnabled(enable);

	glView = newGlView;
}

void qtome::tabChanged(int index)
{
	GLView3D *current = 0;
	if (index >= 0)
	{
		QWidget *w = tabs->currentWidget();
		if (w)
		{
			GLContainer *container = static_cast<GLContainer *>(w);
			if (container)
				current = static_cast<GLView3D *>(container->getWindow());
		}
	}
	viewFocusChanged(current);
}

void qtome::quit()
{
	close();
}

void qtome::view_reset()
{
	if (glView)
	{
		glView->setZoom(0);
		glView->setXTranslation(0);
		glView->setYTranslation(0);
		glView->setZRotation(0);
	}
}

void qtome::view_zoom()
{
	if (glView)
		glView->setMouseMode(GLView3D::MODE_ZOOM);
}

void qtome::view_pan()
{
	if (glView)
		glView->setMouseMode(GLView3D::MODE_PAN);
}

void qtome::view_rotate()
{
	if (glView)
		glView->setMouseMode(GLView3D::MODE_ROTATE);
}
