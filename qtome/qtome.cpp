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
#include "ViewerState.h"

#include <QtCore/QElapsedTimer>
#include <QtCore/QSettings>
#include <QtWidgets/QAction>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QToolBar>

#include <boost/filesystem/path.hpp>

qtome::qtome(QWidget* parent)
  : QMainWindow(parent)
{
  QCoreApplication::setOrganizationName("AICS");
  QCoreApplication::setOrganizationDomain("allencell.org");
  QCoreApplication::setApplicationName("VolumeRenderer");

  m_ui.setupUi(this);

  createActions();
  createMenus();
  createToolbars();
  createDockWindows();
  setDockOptions(AllowTabbedDocks);

  m_tabs = new QTabWidget(this);

  QHBoxLayout* mainLayout = new QHBoxLayout;
  mainLayout->addWidget(m_tabs);

  QWidget* central = new QWidget(this);
  central->setLayout(mainLayout);

  setCentralWidget(central);

  connect(m_tabs, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));

  // add the single gl view as a tab
  m_glView = new GLView3D(&m_qcamera, &m_transferFunction, &m_renderSettings, this);
  QWidget* glContainer = new GLContainer(this, m_glView);
  m_glView->setObjectName("glcontainer");
  // We need a minimum size or else the size defaults to zero.
  glContainer->setMinimumSize(512, 512);
  m_tabs->addTab(glContainer, "None");
  // navigationZCChanged = connect(navigation, SIGNAL(cChanged(size_t)), glView, SLOT(setC(size_t)));

  setWindowTitle(tr("AICS high performance volume viewer"));

  m_appScene.initLights();
}

void
qtome::createActions()
{
  boost::filesystem::path iconpath(QCoreApplication::applicationDirPath().toStdString());

  m_openAction = new QAction(tr("&Open image..."), this);
  m_openAction->setShortcuts(QKeySequence::Open);
  m_openAction->setStatusTip(tr("Open an existing image file"));
  connect(m_openAction, SIGNAL(triggered()), this, SLOT(open()));

  m_openJsonAction = new QAction(tr("&Open json..."), this);
  m_openJsonAction->setShortcuts(QKeySequence::Open);
  m_openJsonAction->setStatusTip(tr("Open an existing json settings file"));
  connect(m_openJsonAction, SIGNAL(triggered()), this, SLOT(openJson()));

  m_quitAction = new QAction(tr("&Quit"), this);
  m_quitAction->setShortcuts(QKeySequence::Quit);
  m_quitAction->setStatusTip(tr("Quit the application"));
  connect(m_quitAction, SIGNAL(triggered()), this, SLOT(quit()));

  m_viewResetAction = new QAction(tr("&Reset"), this);
  m_viewResetAction->setShortcut(QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_R));
  m_viewResetAction->setStatusTip(tr("Reset the current view"));
  QIcon reset_icon(QString((iconpath / "actions/ome-reset2d.svg").string().c_str()));
  m_viewResetAction->setIcon(reset_icon);
  m_viewResetAction->setEnabled(false);
  connect(m_viewResetAction, SIGNAL(triggered()), this, SLOT(view_reset()));

  m_dumpAction = new QAction(tr("&Dump python commands"), this);
  // dumpAction->setShortcuts(QKeySequence::Open);
  m_dumpAction->setStatusTip(tr("Log a string containing a command buffer to paste into python"));
  connect(m_dumpAction, SIGNAL(triggered()), this, SLOT(dumpPythonState()));

  m_dumpJsonAction = new QAction(tr("&Save to json"), this);
  m_dumpJsonAction->setStatusTip(tr("Save a file containing all render settings and loaded volume path"));
  connect(m_dumpJsonAction, SIGNAL(triggered()), this, SLOT(saveJson()));

  m_testMeshAction = new QAction(tr("&Open mesh..."), this);
  // testMeshAction->setShortcuts(QKeySequence::Open);
  m_testMeshAction->setStatusTip(tr("Open a mesh obj file"));
  connect(m_testMeshAction, SIGNAL(triggered()), this, SLOT(openMeshDialog()));
}

void
qtome::createMenus()
{
  m_fileMenu = menuBar()->addMenu(tr("&File"));
  m_fileMenu->addAction(m_openAction);
  m_fileMenu->addAction(m_openJsonAction);
  m_fileMenu->addSeparator();
  m_fileMenu->addSeparator();
  m_fileMenu->addAction(m_dumpJsonAction);
  m_fileMenu->addSeparator();
  m_fileMenu->addAction(m_quitAction);

  QMenu* recentMenu = m_fileMenu->addMenu(tr("Recent..."));
  connect(recentMenu, &QMenu::aboutToShow, this, &qtome::updateRecentFileActions);
  m_recentFileSubMenuAct = recentMenu->menuAction();
  for (int i = 0; i < MaxRecentFiles; ++i) {
    m_recentFileActs[i] = recentMenu->addAction(QString(), this, &qtome::openRecentFile);
    m_recentFileActs[i]->setVisible(false);
  }
  m_recentFileSeparator = m_fileMenu->addSeparator();
  setRecentFilesVisible(qtome::hasRecentFiles());

  m_viewMenu = menuBar()->addMenu(tr("&View"));

  m_fileMenu->addSeparator();
}

void
qtome::createToolbars()
{}

QDockWidget*
qtome::createRenderingDock()
{
  QDockWidget* dock = new QDockWidget(tr("Rendering"), this);
  dock->setAllowedAreas(Qt::AllDockWidgetAreas);

  QGridLayout* layout = new QGridLayout;

  QLabel* minLabel = new QLabel(tr("Min"));
  QLabel* maxLabel = new QLabel(tr("Max"));
  // minSlider = createRangeSlider();
  // maxSlider = createRangeSlider();

  layout->addWidget(minLabel, 0, 0);
  // layout->addWidget(minSlider, 0, 1);
  layout->addWidget(maxLabel, 1, 0);
  // layout->addWidget(maxSlider, 1, 1);

  QWidget* mainWidget = new QWidget(this);
  mainWidget->setLayout(layout);
  dock->setWidget(mainWidget);
  return dock;
}

void
qtome::createDockWindows()
{
  // navigation = new NavigationDock2D(this);
  // navigation->setAllowedAreas(Qt::AllDockWidgetAreas);
  // addDockWidget(Qt::BottomDockWidgetArea, navigation);

  m_cameradock = new QCameraDockWidget(this, &m_qcamera, &m_renderSettings);
  m_cameradock->setAllowedAreas(Qt::AllDockWidgetAreas);
  addDockWidget(Qt::RightDockWidgetArea, m_cameradock);

  m_appearanceDockWidget = new QAppearanceDockWidget(this, &m_transferFunction, &m_renderSettings);
  m_appearanceDockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);
  addDockWidget(Qt::RightDockWidgetArea, m_appearanceDockWidget);

  m_statisticsDockWidget = new QStatisticsDockWidget(this);
  // Statistics dock widget
  m_statisticsDockWidget->setEnabled(true);
  m_statisticsDockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);
  addDockWidget(Qt::RightDockWidgetArea, m_statisticsDockWidget);
  // m_pViewMenu->addAction(m_StatisticsDockWidget.toggleViewAction());

  m_viewMenu->addSeparator();
  m_viewMenu->addAction(m_cameradock->toggleViewAction());
  m_viewMenu->addSeparator();
  m_viewMenu->addAction(m_appearanceDockWidget->toggleViewAction());
  m_viewMenu->addSeparator();
  m_viewMenu->addAction(m_statisticsDockWidget->toggleViewAction());

  //	QDockWidget* dock = createRenderingDock();
  //	addDockWidget(Qt::BottomDockWidgetArea, dock);
  //	viewMenu->addAction(dock->toggleViewAction());
}

QSlider*
qtome::createAngleSlider()
{
  QSlider* slider = new QSlider(Qt::Vertical);
  slider->setRange(0, 365 * 16);
  slider->setSingleStep(16);
  slider->setPageStep(8 * 16);
  slider->setTickInterval(8 * 16);
  slider->setTickPosition(QSlider::TicksRight);
  return slider;
}

QSlider*
qtome::createRangeSlider()
{
  QSlider* slider = new QSlider(Qt::Horizontal);
  slider->setRange(0, 255 * 16);
  slider->setSingleStep(16);
  slider->setPageStep(8 * 16);
  slider->setTickInterval(8 * 16);
  slider->setTickPosition(QSlider::TicksRight);
  return slider;
}

void
qtome::open()
{
  QString file =
    QFileDialog::getOpenFileName(this, tr("Open Image"), QString(), QString(), 0, QFileDialog::DontResolveSymlinks);

  if (!file.isEmpty())
    open(file);
}

void
qtome::openJson()
{
  QString file =
    QFileDialog::getOpenFileName(this, tr("Open Image"), QString(), QString(), 0, QFileDialog::DontResolveSymlinks);

  if (!file.isEmpty()) {
    QFile loadFile(file);
    if (!loadFile.open(QIODevice::ReadOnly)) {
      qWarning("Couldn't open json file.");
      return;
    }
    QByteArray saveData = loadFile.readAll();
    QJsonDocument loadDoc(QJsonDocument::fromJson(saveData));
    ViewerState s;
    s.stateFromJson(loadDoc);
    if (!s.m_volumeImageFile.isEmpty()) {
      open(s.m_volumeImageFile, &s);
    }
  }
}

void
qtome::saveJson()
{
  QString file = QFileDialog::getSaveFileName(this, tr("Save Json"), QString(), tr("json (*.json)"));
  if (!file.isEmpty()) {
    ViewerState st = appToViewerState();
    QJsonDocument doc = st.stateToJson();
    QFile saveFile(file);
    if (!saveFile.open(QIODevice::WriteOnly)) {
      qWarning("Couldn't open save file.");
      return;
    }
    saveFile.write(doc.toJson());
  }
}

void
qtome::open(const QString& file, const ViewerState* vs)
{
  QFileInfo info(file);
  if (info.exists()) {
    LOG_DEBUG << "Attempting to open " << file.toStdString();

    std::shared_ptr<ImageXYZC> image = FileReader::loadOMETiff_4D(file.toStdString());

    // install the new volume image into the scene.
    // this is deref'ing the previous _volume shared_ptr.
    m_appScene.m_volume = image;

    m_appScene.initSceneFromImg(image);
    m_glView->initCameraFromImage(&m_appScene);

    // initialize _appScene from ViewerState
    if (vs) {
      viewerStateToApp(*vs);
    }

    // tell the 3d view to update.
    // it causes a new renderer which owns the CStatus used below
    m_glView->onNewImage(&m_appScene);
    m_tabs->setTabText(0, info.fileName());
    // navigation->setReader(image);

    m_appearanceDockWidget->onNewImage(&m_appScene);

    // set up status view with some stats.
    CStatus* s = m_glView->getStatus();
    m_statisticsDockWidget->setStatus(s);
    s->onNewImage(info.fileName(), &m_appScene);

    m_currentFilePath = file;
    qtome::prependToRecentFiles(file);
  } else {
    LOG_DEBUG << "Failed to open " << file.toStdString();
  }
}

void
qtome::openMeshDialog()
{
  QString file =
    QFileDialog::getOpenFileName(this, tr("Open Mesh"), QString(), QString(), 0, QFileDialog::DontResolveSymlinks);

  if (!file.isEmpty())
    openMesh(file);
}

void
qtome::openMesh(const QString& file)
{
  if (m_appScene.m_volume) {
    return;
  }
  // load obj file and init scene...
  CBoundingBox bb;
  Assimp::Importer* importer = FileReader::loadAsset(file.toStdString().c_str(), &bb);
  if (importer->GetScene()) {
    m_appScene.m_meshes.push_back(std::shared_ptr<Assimp::Importer>(importer));
    m_appScene.initBounds(bb);
    m_renderSettings.m_DirtyFlags.SetFlag(MeshDirty);
    // tell the 3d view to update.
    m_glView->initCameraFromImage(&m_appScene);
    m_glView->onNewImage(&m_appScene);
    m_appearanceDockWidget->onNewImage(&m_appScene);
  }
}

void
qtome::viewFocusChanged(GLView3D* newGlView)
{
  if (m_glView == newGlView)
    return;

  // disconnect(navigationChanged);
  // disconnect(navigationZCChanged);
  // disconnect(navigationUpdate);

  m_viewResetAction->setEnabled(false);

  if (newGlView) {
    // navigation->setReader(newGlView->getImage());
    // navigationZCChanged = connect(navigation, SIGNAL(cChanged(size_t)), newGlView, SLOT(setC(size_t)));

  } else {
    // navigation->setReader(std::shared_ptr<ImageXYZC>());
  }

  bool enable(newGlView != 0);

  m_viewResetAction->setEnabled(enable);

  m_glView = newGlView;
}

void
qtome::tabChanged(int index)
{
  GLView3D* current = 0;
  if (index >= 0) {
    QWidget* w = m_tabs->currentWidget();
    if (w) {
      GLContainer* container = static_cast<GLContainer*>(w);
      if (container)
        current = static_cast<GLView3D*>(container->getWindow());
    }
  }
  viewFocusChanged(current);
}

void
qtome::quit()
{
  close();
}

void
qtome::view_reset()
{}

void
qtome::setRecentFilesVisible(bool visible)
{
  m_recentFileSubMenuAct->setVisible(visible);
  m_recentFileSeparator->setVisible(visible);
}

static inline QString
recentFilesKey()
{
  return QStringLiteral("recentFileList");
}
static inline QString
fileKey()
{
  return QStringLiteral("file");
}

static QStringList
readRecentFiles(QSettings& settings)
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

static void
writeRecentFiles(const QStringList& files, QSettings& settings)
{
  const int count = files.size();
  settings.beginWriteArray(recentFilesKey());
  for (int i = 0; i < count; ++i) {
    settings.setArrayIndex(i);
    settings.setValue(fileKey(), files.at(i));
  }
  settings.endArray();
}

bool
qtome::hasRecentFiles()
{
  QSettings settings(QCoreApplication::organizationName(), QCoreApplication::applicationName());
  const int count = settings.beginReadArray(recentFilesKey());
  settings.endArray();
  return count > 0;
}

void
qtome::prependToRecentFiles(const QString& fileName)
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

void
qtome::updateRecentFileActions()
{
  QSettings settings(QCoreApplication::organizationName(), QCoreApplication::applicationName());

  const QStringList recentFiles = readRecentFiles(settings);
  const int count = qMin(int(MaxRecentFiles), recentFiles.size());
  int i = 0;
  for (; i < count; ++i) {
    const QString fileName = qtome::strippedName(recentFiles.at(i));
    m_recentFileActs[i]->setText(tr("&%1 %2").arg(i + 1).arg(fileName));
    m_recentFileActs[i]->setData(recentFiles.at(i));
    m_recentFileActs[i]->setVisible(true);
  }
  for (; i < MaxRecentFiles; ++i)
    m_recentFileActs[i]->setVisible(false);
}

void
qtome::openRecentFile()
{
  if (const QAction* action = qobject_cast<const QAction*>(sender())) {
    QString path = action->data().toString();
    if (path.endsWith(".obj")) {
      // assume that .obj is mesh
      openMesh(path);
    } else {
      // assumption of ome.tif
      open(path);
    }
  }
}

QString
qtome::strippedName(const QString& fullFileName)
{
  return QFileInfo(fullFileName).fileName();
}

void
qtome::dumpPythonState()
{
  QString s;
  s += "cb = CommandBuffer()\n";
  s += QString("cb.add_command(\"LOAD_OME_TIF\", \"%1\")\n").arg(m_currentFilePath);
  s += QString("cb.add_command(\"SET_RESOLUTION\", %1, %2)\n")
         .arg(m_glView->size().width())
         .arg(m_glView->size().height());
  s += QString("cb.add_command(\"RENDER_ITERATIONS\", %1)\n").arg(m_renderSettings.GetNoIterations());

  s += QString("cb.add_command(\"SET_CLIP_REGION\", %1, %2, %3, %4, %5, %6)\n")
         .arg(m_appScene.m_roi.GetMinP().x)
         .arg(m_appScene.m_roi.GetMaxP().x)
         .arg(m_appScene.m_roi.GetMinP().y)
         .arg(m_appScene.m_roi.GetMaxP().y)
         .arg(m_appScene.m_roi.GetMinP().z)
         .arg(m_appScene.m_roi.GetMaxP().z);

  s += QString("cb.add_command(\"EYE\", %1, %2, %3)\n")
         .arg(m_glView->getCamera().m_From.x)
         .arg(m_glView->getCamera().m_From.y)
         .arg(m_glView->getCamera().m_From.z);
  s += QString("cb.add_command(\"TARGET\", %1, %2, %3)\n")
         .arg(m_glView->getCamera().m_Target.x)
         .arg(m_glView->getCamera().m_Target.y)
         .arg(m_glView->getCamera().m_Target.z);
  s += QString("cb.add_command(\"UP\", %1, %2, %3)\n")
         .arg(m_glView->getCamera().m_Up.x)
         .arg(m_glView->getCamera().m_Up.y)
         .arg(m_glView->getCamera().m_Up.z);
  s += QString("cb.add_command(\"FOV_Y\", %1)\n").arg(m_qcamera.GetProjection().GetFieldOfView());

  s += QString("cb.add_command(\"EXPOSURE\", %1)\n").arg(m_qcamera.GetFilm().GetExposure());
  s += QString("cb.add_command(\"DENSITY\", %1)\n").arg(m_renderSettings.m_RenderSettings.m_DensityScale);
  s += QString("cb.add_command(\"APERTURE\", %1)\n").arg(m_qcamera.GetAperture().GetSize());
  s += QString("cb.add_command(\"FOCALDIST\", %1)\n").arg(m_qcamera.GetFocus().GetFocalDistance());

  // per-channel
  for (uint32_t i = 0; i < m_appScene.m_volume->sizeC(); ++i) {
    bool enabled = m_appScene.m_material.m_enabled[i];
    s += QString("cb.add_command(\"ENABLE_CHANNEL\", %1, %2)\n").arg(QString::number(i), enabled ? "1" : "0");
    s += QString("cb.add_command(\"MAT_DIFFUSE\", %1, %2, %3, %4, 1.0)\n")
           .arg(QString::number(i))
           .arg(m_appScene.m_material.m_diffuse[i * 3])
           .arg(m_appScene.m_material.m_diffuse[i * 3 + 1])
           .arg(m_appScene.m_material.m_diffuse[i * 3 + 2]);
    s += QString("cb.add_command(\"MAT_SPECULAR\", %1, %2, %3, %4, 0.0)\n")
           .arg(QString::number(i))
           .arg(m_appScene.m_material.m_specular[i * 3])
           .arg(m_appScene.m_material.m_specular[i * 3 + 1])
           .arg(m_appScene.m_material.m_specular[i * 3 + 2]);
    s += QString("cb.add_command(\"MAT_EMISSIVE\", %1, %2, %3, %4, 0.0)\n")
           .arg(QString::number(i))
           .arg(m_appScene.m_material.m_emissive[i * 3])
           .arg(m_appScene.m_material.m_emissive[i * 3 + 1])
           .arg(m_appScene.m_material.m_emissive[i * 3 + 2]);
    s += QString("cb.add_command(\"MAT_GLOSSINESS\", %1, %2)\n")
           .arg(QString::number(i))
           .arg(m_appScene.m_material.m_roughness[i]);
    s += QString("cb.add_command(\"SET_WINDOW_LEVEL\", %1, %2, %3)\n")
           .arg(QString::number(i))
           .arg(m_appScene.m_volume->channel(i)->m_window)
           .arg(m_appScene.m_volume->channel(i)->m_level);
  }

  // lighting
  s += QString("cb.add_command(\"SKYLIGHT_TOP_COLOR\", %1, %2, %3)\n")
         .arg(m_appScene.m_lighting.m_Lights[0].m_ColorTop.r)
         .arg(m_appScene.m_lighting.m_Lights[0].m_ColorTop.g)
         .arg(m_appScene.m_lighting.m_Lights[0].m_ColorTop.b);
  s += QString("cb.add_command(\"SKYLIGHT_MIDDLE_COLOR\", %1, %2, %3)\n")
         .arg(m_appScene.m_lighting.m_Lights[0].m_ColorMiddle.r)
         .arg(m_appScene.m_lighting.m_Lights[0].m_ColorMiddle.g)
         .arg(m_appScene.m_lighting.m_Lights[0].m_ColorMiddle.b);
  s += QString("cb.add_command(\"SKYLIGHT_BOTTOM_COLOR\", %1, %2, %3)\n")
         .arg(m_appScene.m_lighting.m_Lights[0].m_ColorBottom.r)
         .arg(m_appScene.m_lighting.m_Lights[0].m_ColorBottom.g)
         .arg(m_appScene.m_lighting.m_Lights[0].m_ColorBottom.b);
  s += QString("cb.add_command(\"LIGHT_POS\", 0, %1, %2, %3)\n")
         .arg(m_appScene.m_lighting.m_Lights[1].m_Distance)
         .arg(m_appScene.m_lighting.m_Lights[1].m_Theta)
         .arg(m_appScene.m_lighting.m_Lights[1].m_Phi);
  s += QString("cb.add_command(\"LIGHT_COLOR\", 0, %1, %2, %3)\n")
         .arg(m_appScene.m_lighting.m_Lights[1].m_Color.r)
         .arg(m_appScene.m_lighting.m_Lights[1].m_Color.g)
         .arg(m_appScene.m_lighting.m_Lights[1].m_Color.b);
  s += QString("cb.add_command(\"LIGHT_SIZE\", 0, %1, %2)\n")
         .arg(m_appScene.m_lighting.m_Lights[1].m_Width)
         .arg(m_appScene.m_lighting.m_Lights[1].m_Height);

  s += "buf = cb.make_buffer()\n";
  qDebug().noquote() << s;
  // return s;
}

void
qtome::dumpStateToJson()
{
  ViewerState st = appToViewerState();
  QJsonDocument doc = st.stateToJson();
  QString s = doc.toJson();
  qDebug().noquote() << s;
}

void
qtome::viewerStateToApp(const ViewerState& v)
{
  // ASSUME THAT IMAGE IS LOADED AND APPSCENE INITIALIZED

  // position camera
  m_glView->fromViewerState(v);

  m_appScene.m_roi.SetMinP(glm::vec3(v.m_roiXmin, v.m_roiYmin, v.m_roiZmin));
  m_appScene.m_roi.SetMaxP(glm::vec3(v.m_roiXmax, v.m_roiYmax, v.m_roiZmax));

  m_appScene.m_volume->setPhysicalSize(v.m_scaleX, v.m_scaleY, v.m_scaleZ);

  m_renderSettings.m_RenderSettings.m_DensityScale = v.m_densityScale;

  // channels
  for (uint32_t i = 0; i < m_appScene.m_volume->sizeC(); ++i) {
    ChannelViewerState ch = v.m_channels[i];
    m_appScene.m_material.m_enabled[i] = ch.m_enabled;

    m_appScene.m_material.m_diffuse[i * 3] = ch.m_diffuse.x;
    m_appScene.m_material.m_diffuse[i * 3 + 1] = ch.m_diffuse.y;
    m_appScene.m_material.m_diffuse[i * 3 + 2] = ch.m_diffuse.z;

    m_appScene.m_material.m_specular[i * 3] = ch.m_specular.x;
    m_appScene.m_material.m_specular[i * 3 + 1] = ch.m_specular.y;
    m_appScene.m_material.m_specular[i * 3 + 2] = ch.m_specular.z;

    m_appScene.m_material.m_emissive[i * 3] = ch.m_emissive.x;
    m_appScene.m_material.m_emissive[i * 3 + 1] = ch.m_emissive.y;
    m_appScene.m_material.m_emissive[i * 3 + 2] = ch.m_emissive.z;

    m_appScene.m_material.m_roughness[i] = ch.m_glossiness;
    m_appScene.m_volume->channel(i)->generate_windowLevel(ch.m_window, ch.m_level);
  }

  // lights
  Light& lt = m_appScene.m_lighting.m_Lights[0];
  lt.m_T = v.m_light0.m_type;
  lt.m_Distance = v.m_light0.m_distance;
  lt.m_Theta = v.m_light0.m_theta;
  lt.m_Phi = v.m_light0.m_phi;
  lt.m_ColorTop = v.m_light0.m_topColor;
  lt.m_ColorMiddle = v.m_light0.m_middleColor;
  lt.m_ColorBottom = v.m_light0.m_bottomColor;
  lt.m_Color = v.m_light0.m_color;
  lt.m_ColorTopIntensity = v.m_light0.m_topColorIntensity;
  lt.m_ColorMiddleIntensity = v.m_light0.m_middleColorIntensity;
  lt.m_ColorBottomIntensity = v.m_light0.m_bottomColorIntensity;
  lt.m_ColorIntensity = v.m_light0.m_colorIntensity;
  lt.m_Width = v.m_light0.m_width;
  lt.m_Height = v.m_light0.m_height;

  Light& lt1 = m_appScene.m_lighting.m_Lights[1];
  lt1.m_T = v.m_light1.m_type;
  lt1.m_Distance = v.m_light1.m_distance;
  lt1.m_Theta = v.m_light1.m_theta;
  lt1.m_Phi = v.m_light1.m_phi;
  lt1.m_ColorTop = v.m_light1.m_topColor;
  lt1.m_ColorMiddle = v.m_light1.m_middleColor;
  lt1.m_ColorBottom = v.m_light1.m_bottomColor;
  lt1.m_Color = v.m_light1.m_color;
  lt1.m_ColorTopIntensity = v.m_light1.m_topColorIntensity;
  lt1.m_ColorMiddleIntensity = v.m_light1.m_middleColorIntensity;
  lt1.m_ColorBottomIntensity = v.m_light1.m_bottomColorIntensity;
  lt1.m_ColorIntensity = v.m_light1.m_colorIntensity;
  lt1.m_Width = v.m_light1.m_width;
  lt1.m_Height = v.m_light1.m_height;

  m_renderSettings.m_DirtyFlags.SetFlag(RenderParamsDirty);
}

ViewerState
qtome::appToViewerState()
{
  ViewerState v;
  v.m_volumeImageFile = m_currentFilePath;

  v.m_scaleX = m_appScene.m_volume->physicalSizeX();
  v.m_scaleY = m_appScene.m_volume->physicalSizeY();
  v.m_scaleZ = m_appScene.m_volume->physicalSizeZ();

  v.m_resolutionX = m_glView->size().width();
  v.m_resolutionY = m_glView->size().height();
  v.m_renderIterations = m_renderSettings.GetNoIterations();

  v.m_roiXmax = m_appScene.m_roi.GetMaxP().x;
  v.m_roiYmax = m_appScene.m_roi.GetMaxP().y;
  v.m_roiZmax = m_appScene.m_roi.GetMaxP().z;
  v.m_roiXmin = m_appScene.m_roi.GetMinP().x;
  v.m_roiYmin = m_appScene.m_roi.GetMinP().y;
  v.m_roiZmin = m_appScene.m_roi.GetMinP().z;

  v.m_eyeX = m_glView->getCamera().m_From.x;
  v.m_eyeY = m_glView->getCamera().m_From.y;
  v.m_eyeZ = m_glView->getCamera().m_From.z;

  v.m_targetX = m_glView->getCamera().m_Target.x;
  v.m_targetY = m_glView->getCamera().m_Target.y;
  v.m_targetZ = m_glView->getCamera().m_Target.z;

  v.m_upX = m_glView->getCamera().m_Up.x;
  v.m_upY = m_glView->getCamera().m_Up.y;
  v.m_upZ = m_glView->getCamera().m_Up.z;

  v.m_fov = m_qcamera.GetProjection().GetFieldOfView();

  v.m_exposure = m_qcamera.GetFilm().GetExposure();
  v.m_apertureSize = m_qcamera.GetAperture().GetSize();
  v.m_focalDistance = m_qcamera.GetFocus().GetFocalDistance();
  v.m_densityScale = m_renderSettings.m_RenderSettings.m_DensityScale;

  for (uint32_t i = 0; i < m_appScene.m_volume->sizeC(); ++i) {
    ChannelViewerState ch;
    ch.m_enabled = m_appScene.m_material.m_enabled[i];
    ch.m_diffuse = glm::vec3(m_appScene.m_material.m_diffuse[i * 3],
                             m_appScene.m_material.m_diffuse[i * 3 + 1],
                             m_appScene.m_material.m_diffuse[i * 3 + 2]);
    ch.m_specular = glm::vec3(m_appScene.m_material.m_specular[i * 3],
                              m_appScene.m_material.m_specular[i * 3 + 1],
                              m_appScene.m_material.m_specular[i * 3 + 2]);
    ch.m_emissive = glm::vec3(m_appScene.m_material.m_emissive[i * 3],
                              m_appScene.m_material.m_emissive[i * 3 + 1],
                              m_appScene.m_material.m_emissive[i * 3 + 2]);
    ch.m_glossiness = m_appScene.m_material.m_roughness[i];
    ch.m_window = m_appScene.m_volume->channel(i)->m_window;
    ch.m_level = m_appScene.m_volume->channel(i)->m_level;

    v.m_channels.push_back(ch);
  }

  // lighting
  Light& lt = m_appScene.m_lighting.m_Lights[0];
  v.m_light0.m_type = lt.m_T;
  v.m_light0.m_distance = lt.m_Distance;
  v.m_light0.m_theta = lt.m_Theta;
  v.m_light0.m_phi = lt.m_Phi;
  v.m_light0.m_topColor = glm::vec3(lt.m_ColorTop.r, lt.m_ColorTop.g, lt.m_ColorTop.b);
  v.m_light0.m_middleColor = glm::vec3(lt.m_ColorMiddle.r, lt.m_ColorMiddle.g, lt.m_ColorMiddle.b);
  v.m_light0.m_color = glm::vec3(lt.m_Color.r, lt.m_Color.g, lt.m_Color.b);
  v.m_light0.m_bottomColor = glm::vec3(lt.m_ColorBottom.r, lt.m_ColorBottom.g, lt.m_ColorBottom.b);
  v.m_light0.m_topColorIntensity = lt.m_ColorTopIntensity;
  v.m_light0.m_middleColorIntensity = lt.m_ColorMiddleIntensity;
  v.m_light0.m_colorIntensity = lt.m_ColorIntensity;
  v.m_light0.m_bottomColorIntensity = lt.m_ColorBottomIntensity;
  v.m_light0.m_width = lt.m_Width;
  v.m_light0.m_height = lt.m_Height;

  Light& lt1 = m_appScene.m_lighting.m_Lights[1];
  v.m_light1.m_type = lt1.m_T;
  v.m_light1.m_distance = lt1.m_Distance;
  v.m_light1.m_theta = lt1.m_Theta;
  v.m_light1.m_phi = lt1.m_Phi;
  v.m_light1.m_topColor = glm::vec3(lt1.m_ColorTop.r, lt1.m_ColorTop.g, lt1.m_ColorTop.b);
  v.m_light1.m_middleColor = glm::vec3(lt1.m_ColorMiddle.r, lt1.m_ColorMiddle.g, lt1.m_ColorMiddle.b);
  v.m_light1.m_color = glm::vec3(lt1.m_Color.r, lt1.m_Color.g, lt1.m_Color.b);
  v.m_light1.m_bottomColor = glm::vec3(lt1.m_ColorBottom.r, lt1.m_ColorBottom.g, lt1.m_ColorBottom.b);
  v.m_light1.m_topColorIntensity = lt1.m_ColorTopIntensity;
  v.m_light1.m_middleColorIntensity = lt1.m_ColorMiddleIntensity;
  v.m_light1.m_colorIntensity = lt1.m_ColorIntensity;
  v.m_light1.m_bottomColorIntensity = lt1.m_ColorBottomIntensity;
  v.m_light1.m_width = lt1.m_Width;
  v.m_light1.m_height = lt1.m_Height;

  return v;
}
