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


	setWindowTitle(tr("AICS high performance volume viewer"));

	_appScene.initLights();
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

	dumpAction = new QAction(tr("&Dump python commands"), this);
	//dumpAction->setShortcuts(QKeySequence::Open);
	dumpAction->setStatusTip(tr("Log a string containing a command buffer to paste into python"));
	connect(dumpAction, SIGNAL(triggered()), this, SLOT(dumpPythonState()));

	dumpJsonAction = new QAction(tr("&Dump json obj"), this);
	dumpJsonAction->setStatusTip(tr("Log a string containing a json object"));
	connect(dumpJsonAction, SIGNAL(triggered()), this, SLOT(dumpStateToJson()));

	testMeshAction = new QAction(tr("&Open mesh..."), this);
	//testMeshAction->setShortcuts(QKeySequence::Open);
	testMeshAction->setStatusTip(tr("Open a mesh obj file"));
	connect(testMeshAction, SIGNAL(triggered()), this, SLOT(openMeshDialog()));
}

void qtome::createMenus()
{
	fileMenu = menuBar()->addMenu(tr("&File"));
	fileMenu->addAction(openAction);
	fileMenu->addSeparator();
	fileMenu->addAction(testMeshAction);
	fileMenu->addSeparator();
	fileMenu->addAction(dumpAction);
	fileMenu->addAction(dumpJsonAction);
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
		LOG_DEBUG << "Loaded " << file.toStdString() << " in " << t.elapsed() << "ms";

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

		_currentFilePath = file;
		qtome::prependToRecentFiles(file);
	}



}

void qtome::openMeshDialog() {
	QString file = QFileDialog::getOpenFileName(this,
		tr("Open Mesh"),
		QString(),
		QString(),
		0,
		QFileDialog::DontResolveSymlinks);

	if (!file.isEmpty())
		openMesh(file);
}

void qtome::openMesh(const QString& file) {
	if (_appScene._volume) {
		return;
	}
	// load obj file and init scene...
	CBoundingBox bb;
	FileReader fileReader;
	QElapsedTimer t;
	t.start();
	//Assimp::Importer* importer = fileReader.loadAsset("C:\\Users\\danielt.ALLENINST\\Downloads\\nucleus.obj", &bb);
	Assimp::Importer* importer = fileReader.loadAsset(file.toStdString().c_str(), &bb);
	if (importer->GetScene()) {
		_appScene._meshes.push_back(std::shared_ptr<Assimp::Importer>(importer));
		_appScene.initSceneFromBoundingBox(bb);
		_renderSettings.m_DirtyFlags.SetFlag(MeshDirty);
		// tell the 3d view to update.
		glView->onNewImage(&_appScene);
		appearanceDockWidget->onNewImage(&_appScene);
	}
	LOG_DEBUG << "Loaded mesh in " << t.elapsed() << "ms";
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
	if (const QAction *action = qobject_cast<const QAction *>(sender())) {
		QString path = action->data().toString();
		if (path.endsWith(".obj")) {
			// assume that .obj is mesh
			openMesh(path);
		}
		else {
			// assumption of ome.tif
			open(path);
		}
	}
}

QString qtome::strippedName(const QString &fullFileName)
{
	return QFileInfo(fullFileName).fileName();
}

void qtome::dumpPythonState()
{
	QString s;
	s += "cb = CommandBuffer()\n";
	s += QString("cb.add_command(\"LOAD_OME_TIF\", \"%1\")\n").arg(_currentFilePath);
	s += QString("cb.add_command(\"SET_RESOLUTION\", %1, %2)\n").arg(glView->size().width()).arg(glView->size().height());
	s += QString("cb.add_command(\"RENDER_ITERATIONS\", %1)\n").arg(_renderSettings.GetNoIterations());

	s += QString("cb.add_command(\"SET_CLIP_REGION\", %1, %2, %3, %4, %5, %6)\n").arg(_appScene._roi.GetMinP().x).arg(_appScene._roi.GetMaxP().x).arg(_appScene._roi.GetMinP().y).arg(_appScene._roi.GetMaxP().y).arg(_appScene._roi.GetMinP().z).arg(_appScene._roi.GetMaxP().z);

	s += QString("cb.add_command(\"EYE\", %1, %2, %3)\n").arg(glView->getCamera().m_From.x).arg(glView->getCamera().m_From.y).arg(glView->getCamera().m_From.z);
	s += QString("cb.add_command(\"TARGET\", %1, %2, %3)\n").arg(glView->getCamera().m_Target.x).arg(glView->getCamera().m_Target.y).arg(glView->getCamera().m_Target.z);
	s += QString("cb.add_command(\"UP\", %1, %2, %3)\n").arg(glView->getCamera().m_Up.x).arg(glView->getCamera().m_Up.y).arg(glView->getCamera().m_Up.z);
	s += QString("cb.add_command(\"FOV_Y\", %1)\n").arg(_camera.GetProjection().GetFieldOfView());
	
	s += QString("cb.add_command(\"EXPOSURE\", %1)\n").arg(_camera.GetFilm().GetExposure());
	s += QString("cb.add_command(\"DENSITY\", %1)\n").arg(_renderSettings.m_RenderSettings.m_DensityScale);
	s += QString("cb.add_command(\"APERTURE\", %1)\n").arg(_camera.GetAperture().GetSize());
	s += QString("cb.add_command(\"FOCALDIST\", %1)\n").arg(_camera.GetFocus().GetFocalDistance());

	// per-channel
	for (uint32_t i = 0; i < _appScene._volume->sizeC(); ++i) {
		bool enabled = _appScene._material.enabled[i];
		s += QString("cb.add_command(\"ENABLE_CHANNEL\", %1, %2)\n").arg(QString::number(i), enabled?"1":"0");
		s += QString("cb.add_command(\"MAT_DIFFUSE\", %1, %2, %3, %4, 1.0)\n").arg(QString::number(i)).arg(_appScene._material.diffuse[i*3]).arg(_appScene._material.diffuse[i * 3+1]).arg(_appScene._material.diffuse[i * 3+2]);
		s += QString("cb.add_command(\"MAT_SPECULAR\", %1, %2, %3, %4, 0.0)\n").arg(QString::number(i)).arg(_appScene._material.specular[i * 3]).arg(_appScene._material.specular[i * 3 + 1]).arg(_appScene._material.specular[i * 3 + 2]);
		s += QString("cb.add_command(\"MAT_EMISSIVE\", %1, %2, %3, %4, 0.0)\n").arg(QString::number(i)).arg(_appScene._material.emissive[i * 3]).arg(_appScene._material.emissive[i * 3 + 1]).arg(_appScene._material.emissive[i * 3 + 2]);
		s += QString("cb.add_command(\"MAT_GLOSSINESS\", %1, %2)\n").arg(QString::number(i)).arg(_appScene._material.roughness[i]);
		s += QString("cb.add_command(\"SET_WINDOW_LEVEL\", %1, %2, %3)\n").arg(QString::number(i)).arg(_appScene._volume->channel(i)->_window).arg(_appScene._volume->channel(i)->_level);
	}

	// lighting
	s += QString("cb.add_command(\"SKYLIGHT_TOP_COLOR\", %1, %2, %3)\n").arg(_appScene._lighting.m_Lights[0].m_ColorTop.r).arg(_appScene._lighting.m_Lights[0].m_ColorTop.g).arg(_appScene._lighting.m_Lights[0].m_ColorTop.b);
	s += QString("cb.add_command(\"SKYLIGHT_MIDDLE_COLOR\", %1, %2, %3)\n").arg(_appScene._lighting.m_Lights[0].m_ColorMiddle.r).arg(_appScene._lighting.m_Lights[0].m_ColorMiddle.g).arg(_appScene._lighting.m_Lights[0].m_ColorMiddle.b);
	s += QString("cb.add_command(\"SKYLIGHT_BOTTOM_COLOR\", %1, %2, %3)\n").arg(_appScene._lighting.m_Lights[0].m_ColorBottom.r).arg(_appScene._lighting.m_Lights[0].m_ColorBottom.g).arg(_appScene._lighting.m_Lights[0].m_ColorBottom.b);
	s += QString("cb.add_command(\"LIGHT_POS\", 0, %1, %2, %3)\n").arg(_appScene._lighting.m_Lights[1].m_Distance).arg(_appScene._lighting.m_Lights[1].m_Theta).arg(_appScene._lighting.m_Lights[1].m_Phi);
	s += QString("cb.add_command(\"LIGHT_COLOR\", 0, %1, %2, %3)\n").arg(_appScene._lighting.m_Lights[1].m_Color.r).arg(_appScene._lighting.m_Lights[1].m_Color.g).arg(_appScene._lighting.m_Lights[1].m_Color.b);
	s += QString("cb.add_command(\"LIGHT_SIZE\", 0, %1, %2)\n").arg(_appScene._lighting.m_Lights[1].m_Width).arg(_appScene._lighting.m_Lights[1].m_Height);

	s += "buf = cb.make_buffer()\n";
	qDebug().noquote() << s;
	//return s;
}

QJsonArray jsonVec3(float x, float y, float z) {
	QJsonArray tgt;
	tgt.append(x);
	tgt.append(y);
	tgt.append(z);
	return tgt;
}

void qtome::dumpStateToJson() {
	QJsonDocument doc = stateToJson();
	QString s = doc.toJson();
	qDebug().noquote() << s;
}

QJsonDocument qtome::stateToJson()
{
	// fire back some json...
	QJsonObject j;
	j["name"] = _currentFilePath;
	
	QJsonArray resolution;
	resolution.append(glView->size().width());
	resolution.append(glView->size().height());
	j["resolution"] = resolution;

	j["renderIterations"] = _renderSettings.GetNoIterations();

	QJsonArray clipRegion;
	QJsonArray clipRegionX;
	clipRegionX.append(_appScene._roi.GetMinP().x);
	clipRegionX.append(_appScene._roi.GetMaxP().x);
	QJsonArray clipRegionY;
	clipRegionY.append(_appScene._roi.GetMinP().y);
	clipRegionY.append(_appScene._roi.GetMaxP().y);
	QJsonArray clipRegionZ;
	clipRegionZ.append(_appScene._roi.GetMinP().z);
	clipRegionZ.append(_appScene._roi.GetMaxP().z);
	clipRegion.append(clipRegionX);
	clipRegion.append(clipRegionY);
	clipRegion.append(clipRegionZ);

	j["clipRegion"] = clipRegion;

	QJsonObject camera;
	camera["eye"] = jsonVec3(
		glView->getCamera().m_From.x,
		glView->getCamera().m_From.y,
		glView->getCamera().m_From.z
	);
	camera["target"] = jsonVec3(
		glView->getCamera().m_Target.x,
		glView->getCamera().m_Target.y,
		glView->getCamera().m_Target.z
	);
	camera["up"] = jsonVec3(
		glView->getCamera().m_Up.x,
		glView->getCamera().m_Up.y,
		glView->getCamera().m_Up.z
	);

	camera["fovY"] = _camera.GetProjection().GetFieldOfView();

	camera["exposure"] = _camera.GetFilm().GetExposure();
	camera["aperture"] = _camera.GetAperture().GetSize();
	camera["focalDistance"] = _camera.GetFocus().GetFocalDistance();
	j["camera"] = camera;

	QJsonArray channels;
	for (uint32_t i = 0; i < _appScene._volume->sizeC(); ++i) {
		QJsonObject channel;
		channel["enabled"] = _appScene._material.enabled[i];
		channel["diffuseColor"] = jsonVec3(
			_appScene._material.diffuse[i * 3],
			_appScene._material.diffuse[i * 3 + 1],
			_appScene._material.diffuse[i * 3 + 2]
		);
		channel["specularColor"] = jsonVec3(
			_appScene._material.specular[i * 3],
			_appScene._material.specular[i * 3 + 1],
			_appScene._material.specular[i * 3 + 2]
		);
		channel["emissiveColor"] = jsonVec3(
			_appScene._material.emissive[i * 3],
			_appScene._material.emissive[i * 3 + 1],
			_appScene._material.emissive[i * 3 + 2]
		);
		channel["glossiness"] = _appScene._material.roughness[i];
		channel["window"] = _appScene._volume->channel(i)->_window;
		channel["level"] = _appScene._volume->channel(i)->_level;

		channels.append(channel);
	}
	j["channels"] = channels;

	j["density"] = _renderSettings.m_RenderSettings.m_DensityScale;

	// lighting
	QJsonArray lights;
	QJsonObject light0;
	Light& lt = _appScene._lighting.m_Lights[0];
	light0["type"] = 0;
	light0["topColor"] = jsonVec3(
		lt.m_ColorTop.r, lt.m_ColorTop.g, lt.m_ColorTop.b
	);
	light0["middleColor"] = jsonVec3(
		lt.m_ColorMiddle.r, lt.m_ColorMiddle.g, lt.m_ColorMiddle.b
	);
	light0["bottomColor"] = jsonVec3(
		lt.m_ColorBottom.r, lt.m_ColorBottom.g, lt.m_ColorBottom.b
	);
	lights.append(light0);

	QJsonObject light1;
	lt = _appScene._lighting.m_Lights[1];
	light1["type"] = 1;
	light1["distance"] = lt.m_Distance;
	light1["theta"] = lt.m_Theta;
	light1["phi"] = lt.m_Phi;
	light1["color"] = jsonVec3(lt.m_Color.r, lt.m_Color.g, lt.m_Color.b);
	light1["width"] = lt.m_Width;
	light1["height"] = lt.m_Height;
	lights.append(light1);
	j["lights"] = lights;

	return QJsonDocument(j);
}
