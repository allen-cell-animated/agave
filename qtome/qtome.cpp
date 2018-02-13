// Include first to avoid clash with Windows headers pulled in via
// QtCore/qt_windows.h; they define VOID and HALFTONE which clash with
// the TIFF enums.
#include <memory>

#include "qtome.h"

#include "renderlib/AppScene.h"
#include "renderlib/FileReader.h"
#include "renderlib/ImageXYZC.h"
#include "renderlib/Logging.h"
#include "renderlib/Status.h"

#include "AppearanceDockWidget.h"
#include "CameraDockWidget.h"
#include "GLContainer.h"
#include "StatisticsDockWidget.h"

#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QAction>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QFileDialog>
#include <QtCore/QElapsedTimer>
#include <QtCore/QSettings>

#include <boost/filesystem/path.hpp>

qtome::qtome(QWidget *parent)
	: QMainWindow(parent)
{
	QCoreApplication::setOrganizationName("AICS");
	QCoreApplication::setOrganizationDomain("allencell.org");
	QCoreApplication::setApplicationName("VolumeRenderer");

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

	// add the single gl view as a tab
	glView = new GLView3D(&_camera, &_transferFunction, &_renderSettings, this);
	QWidget *glContainer = new GLContainer(this, glView);
	glView->setObjectName("glcontainer");
	// We need a minimum size or else the size defaults to zero.
	glContainer->setMinimumSize(512, 512);
	tabs->addTab(glContainer, "None");
	//navigationZCChanged = connect(navigation, SIGNAL(cChanged(size_t)), glView, SLOT(setC(size_t)));


	setWindowTitle(tr("OME-Files GLView"));

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
}

void qtome::createMenus()
{
	fileMenu = menuBar()->addMenu(tr("&File"));
	fileMenu->addAction(openAction);
	fileMenu->addSeparator();
	fileMenu->addAction(quitAction);

	QMenu *recentMenu = fileMenu->addMenu(tr("Recent..."));
	connect(recentMenu, &QMenu::aboutToShow, this, &qtome::updateRecentFileActions);
	recentFileSubMenuAct = recentMenu->menuAction();
	for (int i = 0; i < MaxRecentFiles; ++i) {
		recentFileActs[i] = recentMenu->addAction(QString(), this, &qtome::openRecentFile);
		recentFileActs[i]->setVisible(false);
	}
	recentFileSeparator = fileMenu->addSeparator();
	setRecentFilesVisible(qtome::hasRecentFiles());

	viewMenu = menuBar()->addMenu(tr("&View"));
	viewMenu->addAction(viewResetAction);
	fileMenu->addSeparator();
}

void qtome::createToolbars()
{
	Cam2DTools = new QToolBar("2D Camera", this);
	addToolBar(Qt::TopToolBarArea, Cam2DTools);
	Cam2DTools->addAction(viewResetAction);

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
	//navigation = new NavigationDock2D(this);
	//navigation->setAllowedAreas(Qt::AllDockWidgetAreas);
	//addDockWidget(Qt::BottomDockWidgetArea, navigation);

	cameradock = new QCameraDockWidget(this, &_camera, &_renderSettings);
	cameradock->setAllowedAreas(Qt::AllDockWidgetAreas);
	addDockWidget(Qt::RightDockWidgetArea, cameradock);

	appearanceDockWidget = new QAppearanceDockWidget(this, &_transferFunction, &_renderSettings);
	appearanceDockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);
	addDockWidget(Qt::RightDockWidgetArea, appearanceDockWidget);

	statisticsDockWidget = new QStatisticsDockWidget(this);
	// Statistics dock widget
	statisticsDockWidget->setEnabled(true);
	statisticsDockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);
	addDockWidget(Qt::RightDockWidgetArea, statisticsDockWidget);
	//m_pViewMenu->addAction(m_StatisticsDockWidget.toggleViewAction());


	viewMenu->addSeparator();
	viewMenu->addAction(appearanceDockWidget->toggleViewAction());

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

inline QString FormatVector(const glm::vec3& Vector, const int& Precision = 2)
{
	return "[" + QString::number(Vector.x, 'f', Precision) + ", " + QString::number(Vector.y, 'f', Precision) + ", " + QString::number(Vector.z, 'f', Precision) + "]";
}

inline QString FormatVector(const glm::ivec3& Vector)
{
	return "[" + QString::number(Vector.x) + ", " + QString::number(Vector.y) + ", " + QString::number(Vector.z) + "]";
}

inline QString FormatSize(const glm::vec3& Size, const int& Precision = 2)
{
	return QString::number(Size.x, 'f', Precision) + " x " + QString::number(Size.y, 'f', Precision) + " x " + QString::number(Size.z, 'f', Precision);
}

inline QString FormatSize(const glm::ivec3& Size)
{
	return QString::number(Size.x) + " x " + QString::number(Size.y) + " x " + QString::number(Size.z);
}

void qtome::open(const QString& file)
{
	QFileInfo info(file);
	if (info.exists())
	{

		FileReader fileReader;
		QElapsedTimer t;
		t.start();
		std::shared_ptr<ImageXYZC> image = fileReader.loadOMETiff_4D(file.toStdString());
		qDebug() << "Loaded " << file << " in " << t.elapsed() << "ms";

		// install the new volume image into the scene.
		// this is deref'ing the previous _volume shared_ptr.
		_appScene._volume = image;
		_appScene.initSceneFromImg(image);

		// tell the 3d view to update.
		// it causes a new renderer which owns the CStatus used below
		glView->onNewImage(&_appScene);
		tabs->setTabText(0, info.fileName());
		//navigation->setReader(image);

		appearanceDockWidget->onNewImage(&_appScene);

		CStatus* s = glView->getStatus();
		statisticsDockWidget->setStatus(s);

		glm::vec3 resolution(image->sizeX(), image->sizeY(), image->sizeZ());
		glm::vec3 spacing(image->physicalSizeX(), image->physicalSizeY(), image->physicalSizeZ());
		const glm::vec3 PhysicalSize(
			spacing.x * (float)resolution.x,
			spacing.y * (float)resolution.y,
			spacing.z * (float)resolution.z
		);
		glm::vec3 BoundingBoxMinP = glm::vec3(0.0f);
		glm::vec3 BoundingBoxMaxP = PhysicalSize / std::max(PhysicalSize.x, std::max(PhysicalSize.y, PhysicalSize.z));
		s->SetStatisticChanged("Volume", "File", info.fileName(), "");
		s->SetStatisticChanged("Volume", "Bounding Box", "", "");
		s->SetStatisticChanged("Bounding Box", "Min", FormatVector(BoundingBoxMinP, 2), "m");
		s->SetStatisticChanged("Bounding Box", "Max", FormatVector(BoundingBoxMaxP, 2), "m");
		s->SetStatisticChanged("Volume", "Physical Size", FormatSize(PhysicalSize, 2), "mm");
		s->SetStatisticChanged("Volume", "Resolution", FormatSize(resolution), "Voxels");
		s->SetStatisticChanged("Volume", "Spacing", FormatSize(spacing, 2), "mm");
		s->SetStatisticChanged("Volume", "No. Voxels", QString::number(resolution.x*resolution.y*resolution.z), "Voxels");
		// TODO: this is per channel
		//s->SetStatisticChanged("Volume", "Density Range", "[" + QString::number(gScene.m_IntensityRange.GetMin()) + ", " + QString::number(gScene.m_IntensityRange.GetMax()) + "]", "");

		qtome::prependToRecentFiles(file);
	}
}

void qtome::viewFocusChanged(GLView3D *newGlView)
{
	if (glView == newGlView)
		return;
	
	//disconnect(navigationChanged);
	//disconnect(navigationZCChanged);
	//disconnect(navigationUpdate);

	viewResetAction->setEnabled(false);

	if (newGlView)
	{
		//navigation->setReader(newGlView->getImage());
		//navigationZCChanged = connect(navigation, SIGNAL(cChanged(size_t)), newGlView, SLOT(setC(size_t)));

	}
	else
	{
		//navigation->setReader(std::shared_ptr<ImageXYZC>());
	}

	bool enable(newGlView != 0);

	viewResetAction->setEnabled(enable);

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
}

void qtome::setRecentFilesVisible(bool visible)
{
	recentFileSubMenuAct->setVisible(visible);
	recentFileSeparator->setVisible(visible);
}

static inline QString recentFilesKey() { return QStringLiteral("recentFileList"); }
static inline QString fileKey() { return QStringLiteral("file"); }

static QStringList readRecentFiles(QSettings &settings)
{
	QStringList result;
	const int count = settings.beginReadArray(recentFilesKey());
	for (int i = 0; i < count; ++i) {
		settings.setArrayIndex(i);
		result.append(settings.value(fileKey()).toString());
	}
	settings.endArray();
	return result;
}

static void writeRecentFiles(const QStringList &files, QSettings &settings)
{
	const int count = files.size();
	settings.beginWriteArray(recentFilesKey());
	for (int i = 0; i < count; ++i) {
		settings.setArrayIndex(i);
		settings.setValue(fileKey(), files.at(i));
	}
	settings.endArray();
}

bool qtome::hasRecentFiles()
{
	QSettings settings(QCoreApplication::organizationName(), QCoreApplication::applicationName());
	const int count = settings.beginReadArray(recentFilesKey());
	settings.endArray();
	return count > 0;
}

void qtome::prependToRecentFiles(const QString &fileName)
{
	QSettings settings(QCoreApplication::organizationName(), QCoreApplication::applicationName());

	const QStringList oldRecentFiles = readRecentFiles(settings);
	QStringList recentFiles = oldRecentFiles;
	recentFiles.removeAll(fileName);
	recentFiles.prepend(fileName);
	if (oldRecentFiles != recentFiles)
		writeRecentFiles(recentFiles, settings);

	setRecentFilesVisible(!recentFiles.isEmpty());
}

void qtome::updateRecentFileActions()
{
	QSettings settings(QCoreApplication::organizationName(), QCoreApplication::applicationName());

	const QStringList recentFiles = readRecentFiles(settings);
	const int count = qMin(int(MaxRecentFiles), recentFiles.size());
	int i = 0;
	for (; i < count; ++i) {
		const QString fileName = qtome::strippedName(recentFiles.at(i));
		recentFileActs[i]->setText(tr("&%1 %2").arg(i + 1).arg(fileName));
		recentFileActs[i]->setData(recentFiles.at(i));
		recentFileActs[i]->setVisible(true);
	}
	for (; i < MaxRecentFiles; ++i)
		recentFileActs[i]->setVisible(false);
}

void qtome::openRecentFile()
{
	if (const QAction *action = qobject_cast<const QAction *>(sender()))
		open(action->data().toString());
}

QString qtome::strippedName(const QString &fullFileName)
{
	return QFileInfo(fullFileName).fileName();
}
