// Include first to avoid clash with Windows headers pulled in via
// QtCore/qt_windows.h; they define VOID and HALFTONE which clash with
// the TIFF enums.
#include <memory>

#include "agaveGui.h"

#include "renderlib/AppScene.h"
#include "renderlib/FileReader.h"
#include "renderlib/ImageXYZC.h"
#include "renderlib/Logging.h"
#include "renderlib/Status.h"
#include "renderlib/VolumeDimensions.h"

#include "AppearanceDockWidget.h"
#include "CameraDockWidget.h"
#include "StatisticsDockWidget.h"
#include "TimelineDockWidget.h"
#include "ViewerState.h"
#include "loadDialog.h"
#include "renderDialog.h"

#include <QtCore/QElapsedTimer>
#include <QtCore/QSettings>
#include <QtWidgets/QAction>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QToolBar>

#include <filesystem>

agaveGui::agaveGui(QWidget* parent)
  : QMainWindow(parent)
{
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
  m_glView = new GLView3D(&m_qcamera, &m_qrendersettings, &m_renderSettings, this);
  QObject::connect(m_glView, SIGNAL(ChangedRenderer()), this, SLOT(OnUpdateRenderer()));

  m_glView->setObjectName("glcontainer");
  // We need a minimum size or else the size defaults to zero.
  m_glView->setMinimumSize(256, 512);
  m_tabs->addTab(m_glView, "None");

  QString windowTitle = QApplication::instance()->organizationName() + " " +
                        QApplication::instance()->applicationName() + " " +
                        QApplication::instance()->applicationVersion();
  setWindowTitle(windowTitle);

  m_appScene.initLights();

  resize(1280, 720);
}

void
agaveGui::OnUpdateRenderer()
{
  std::shared_ptr<CStatus> s = m_glView->getStatus();
  m_statisticsDockWidget->setStatus(s);
  // s->onNewImage(info.fileName(), &m_appScene);
}

void
agaveGui::createActions()
{
  // TODO ensure a proper title, shortcut, icon, and statustip for every action
  m_openAction = new QAction(tr("&Open volume..."), this);
  m_openAction->setShortcuts(QKeySequence::Open);
  m_openAction->setStatusTip(tr("Open an existing volume file"));
  connect(m_openAction, SIGNAL(triggered()), this, SLOT(open()));

  m_openUrlAction = new QAction(tr("&Open Zarr volume..."), this);
  m_openUrlAction->setShortcuts(QKeySequence::Open);
  m_openUrlAction->setStatusTip(tr("Open an existing volume file in the cloud or from local directory"));
  connect(m_openUrlAction, SIGNAL(triggered()), this, SLOT(openUrl()));

  m_openJsonAction = new QAction(tr("Open JSON..."), this);
  m_openJsonAction->setStatusTip(tr("Open an existing JSON settings file"));
  connect(m_openJsonAction, SIGNAL(triggered()), this, SLOT(openJson()));

  m_quitAction = new QAction(tr("&Quit"), this);
  m_quitAction->setShortcuts(QKeySequence::Quit);
  m_quitAction->setStatusTip(tr("Quit the application"));
  connect(m_quitAction, SIGNAL(triggered()), this, SLOT(quit()));

  m_viewResetAction = new QAction(tr("&Reset"), this);
  m_viewResetAction->setShortcut(QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_R));
  m_viewResetAction->setToolTip(tr("Reset the current view"));
  m_viewResetAction->setStatusTip(tr("Reset the current view"));
  connect(m_viewResetAction, SIGNAL(triggered()), this, SLOT(view_reset()));

  m_dumpJsonAction = new QAction(tr("&Save to JSON"), this);
  m_dumpJsonAction->setStatusTip(tr("Save a file containing all render settings and loaded volume path"));
  connect(m_dumpJsonAction, SIGNAL(triggered()), this, SLOT(saveJson()));

  m_dumpPythonAction = new QAction(tr("&Save to Python script"), this);
  m_dumpPythonAction->setStatusTip(tr("Save a Python script usable with agave_pyclient"));
  connect(m_dumpPythonAction, SIGNAL(triggered()), this, SLOT(savePython()));

  m_testMeshAction = new QAction(tr("&Open mesh..."), this);
  m_testMeshAction->setStatusTip(tr("Open a mesh obj file"));
  connect(m_testMeshAction, SIGNAL(triggered()), this, SLOT(openMeshDialog()));

  m_toggleCameraProjectionAction = new QAction(tr("Persp/Ortho"), this);
  m_toggleCameraProjectionAction->setToolTip(tr("Toggle perspective and orthographic camera projection modes"));
  m_toggleCameraProjectionAction->setStatusTip(tr("Toggle perspective and orthographic camera projection modes"));
  connect(m_toggleCameraProjectionAction, SIGNAL(triggered()), this, SLOT(view_toggleProjection()));

  m_saveImageAction = new QAction(tr("&Quick render..."), this);
  m_saveImageAction->setStatusTip(tr("Save the current render to an image file"));
  connect(m_saveImageAction, SIGNAL(triggered()), this, SLOT(saveImage()));

  m_renderAction = new QAction(tr("&Render..."), this);
  m_renderAction->setStatusTip(tr("Open the render dialog"));
  connect(m_renderAction, SIGNAL(triggered()), this, SLOT(onRenderAction()));
}

void
agaveGui::createMenus()
{
  m_fileMenu = menuBar()->addMenu(tr("&File"));
  m_fileMenu->addAction(m_openAction);
  m_fileMenu->addAction(m_openUrlAction);
  m_fileMenu->addAction(m_openJsonAction);
  m_fileMenu->addSeparator();
  m_fileMenu->addSeparator();
  m_fileMenu->addAction(m_saveImageAction);
  m_fileMenu->addAction(m_renderAction);
  m_fileMenu->addAction(m_dumpJsonAction);
  m_fileMenu->addAction(m_dumpPythonAction);
  m_fileMenu->addSeparator();
  m_fileMenu->addAction(m_quitAction);

  QMenu* recentMenu = m_fileMenu->addMenu(tr("Recent..."));
  connect(recentMenu, &QMenu::aboutToShow, this, &agaveGui::updateRecentFileActions);
  m_recentFileSubMenuAct = recentMenu->menuAction();
  for (int i = 0; i < MaxRecentFiles; ++i) {
    m_recentFileActs[i] = recentMenu->addAction(QString(), this, &agaveGui::openRecentFile);
    m_recentFileActs[i]->setVisible(false);
  }
  m_recentFileSeparator = m_fileMenu->addSeparator();
  setRecentFilesVisible(agaveGui::hasRecentFiles());

  m_viewMenu = menuBar()->addMenu(tr("&View"));

  m_fileMenu->addSeparator();
}

void
agaveGui::createToolbars()
{
  m_ui.mainToolBar->addAction(m_openAction);
  m_ui.mainToolBar->addAction(m_openUrlAction);
  m_ui.mainToolBar->addAction(m_openJsonAction);
  m_ui.mainToolBar->addSeparator();
  m_ui.mainToolBar->addAction(m_dumpJsonAction);
  m_ui.mainToolBar->addAction(m_dumpPythonAction);
  m_ui.mainToolBar->addAction(m_saveImageAction);
  m_ui.mainToolBar->addAction(m_renderAction);
  m_ui.mainToolBar->addSeparator();
  m_ui.mainToolBar->addAction(m_viewResetAction);
  m_ui.mainToolBar->addAction(m_toggleCameraProjectionAction);
}

void
agaveGui::createDockWindows()
{
  m_cameradock = new QCameraDockWidget(this, &m_qcamera, &m_renderSettings);
  m_cameradock->setAllowedAreas(Qt::AllDockWidgetAreas);
  addDockWidget(Qt::RightDockWidgetArea, m_cameradock);

  m_timelinedock = new QTimelineDockWidget(this, &m_qrendersettings);
  m_timelinedock->setAllowedAreas(Qt::AllDockWidgetAreas);
  addDockWidget(Qt::RightDockWidgetArea, m_timelinedock);

  m_appearanceDockWidget = new QAppearanceDockWidget(this, &m_qrendersettings, &m_renderSettings);
  m_appearanceDockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);
  addDockWidget(Qt::LeftDockWidgetArea, m_appearanceDockWidget);

  m_statisticsDockWidget = new QStatisticsDockWidget(this);
  // Statistics dock widget
  m_statisticsDockWidget->setEnabled(true);
  m_statisticsDockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);
  addDockWidget(Qt::RightDockWidgetArea, m_statisticsDockWidget);

  m_viewMenu->addSeparator();
  m_viewMenu->addAction(m_cameradock->toggleViewAction());
  m_viewMenu->addSeparator();
  m_viewMenu->addAction(m_timelinedock->toggleViewAction());
  m_viewMenu->addSeparator();
  m_viewMenu->addAction(m_appearanceDockWidget->toggleViewAction());
  m_viewMenu->addSeparator();
  m_viewMenu->addAction(m_statisticsDockWidget->toggleViewAction());
}

QSlider*
agaveGui::createAngleSlider()
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
agaveGui::createRangeSlider()
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
agaveGui::open()
{
  QString dir = readRecentDirectory();

  QFileDialog::Options options = QFileDialog::DontResolveSymlinks;
#ifdef __linux__
  options |= QFileDialog::DontUseNativeDialog;
#endif
  QString file = QFileDialog::getOpenFileName(this, tr("Open Volume"), dir, QString(), 0, options);

  if (!file.isEmpty()) {
    if (!open(file.toStdString())) {
      showOpenFailedMessageBox(file);
    }
  }
}

bool
agaveGui::openUrl()
{
  std::string urlToLoad = "";
  bool ok = false;
  QString text =
    QInputDialog::getText(this, tr("Zarr Location"), tr("Enter URL or directory"), QLineEdit::Normal, "", &ok);
  if (ok && !text.isEmpty()) {
    urlToLoad = text.toStdString();
  } else {
    LOG_DEBUG << "Canceled load Zarr from url or directory.";
    return false;
  }

  return open(urlToLoad);
}

void
agaveGui::openJson()
{
  QString dir = readRecentDirectory();

  QFileDialog::Options options = QFileDialog::DontResolveSymlinks;
#ifdef __linux__
  options |= QFileDialog::DontUseNativeDialog;
#endif
  QString file = QFileDialog::getOpenFileName(this, tr("Open JSON"), dir, QString(), 0, options);

  if (!file.isEmpty()) {
    QFile loadFile(file);
    if (!loadFile.open(QIODevice::ReadOnly)) {
      qWarning("Couldn't open JSON file.");
      return;
    }
    QByteArray saveData = loadFile.readAll();
    QJsonDocument loadDoc(QJsonDocument::fromJson(saveData));
    if (loadDoc.isNull()) {
      LOG_DEBUG << "Invalid config file format. Make sure it is JSON.";
      return;
    }

    ViewerState s;
    s.stateFromJson(loadDoc);
    if (!s.m_volumeImageFile.empty()) {
      if (!open(s.m_volumeImageFile, &s)) {
        showOpenFailedMessageBox(file);
      }
    }
  }
}

void
agaveGui::saveImage()
{
  QFileDialog::Options options;
#ifdef __linux__
  options |= QFileDialog::DontUseNativeDialog;
#endif

  const QByteArrayList supportedFormats = QImageWriter::supportedImageFormats();
  QStringList supportedFormatStrings;
  foreach (const QByteArray& item, supportedFormats) {
    supportedFormatStrings.append(QString::fromLocal8Bit(item)); // Assuming local 8-bit.
  }

  static const QStringList desiredFormats = { "png", "jpg", "tif" };

  QStringList formatFilters;
  foreach (const QString& desiredFormatName, desiredFormats) {
    if (supportedFormatStrings.contains(desiredFormatName, Qt::CaseInsensitive)) {
      formatFilters.append(desiredFormatName.toUpper() + " (*." + desiredFormatName + ")");
    }
  }
  QString allSupportedFormatsFilter = formatFilters.join(";;");

  QString file =
    QFileDialog::getSaveFileName(this, tr("Save Image"), QString(), allSupportedFormatsFilter, nullptr, options);
  if (!file.isEmpty()) {
    // capture the viewport
    QImage im = m_glView->captureQimage();
    im.save(file);
  }
}

void
agaveGui::onRenderAction()
{
  // if we are disabling the 3d view then might consider just making this modal
  m_glView->pauseRenderLoop();
  QImage im = m_glView->captureQimage();
  QImage* imcopy = new QImage(im);
  m_glView->doneCurrent();
  m_glView->setEnabled(false);
  m_glView->setUpdatesEnabled(false);
  // extract Renderer from GLView3D to hand to RenderDialog
  IRenderWindow* renderer = m_glView->borrowRenderer();
  if (m_captureSettings.width == 0 && m_captureSettings.height == 0) {
    m_captureSettings.width = imcopy->width();
    m_captureSettings.height = imcopy->height();
  }
  // copy of camera
  CCamera camera = m_glView->getCamera();
  RenderDialog* rdialog = new RenderDialog(
    renderer, m_renderSettings, m_appScene, camera, m_glView->context(), m_loadSpec, &m_captureSettings, this);
  rdialog->resize(geometry().width(), m_tabs->height());
  connect(rdialog, &QDialog::finished, this, [this, &rdialog](int result) {
    // get renderer from RenderDialog and hand it back to GLView3D
    LOG_DEBUG << "RenderDialog finished with result " << result;
    m_glView->setEnabled(true);
    m_glView->resizeGL(m_glView->width(), m_glView->height());
    m_glView->setUpdatesEnabled(true);
    m_glView->restartRenderLoop();
  });

  rdialog->setImage(imcopy);
  delete imcopy;

  rdialog->show();
  rdialog->raise();
  rdialog->activateWindow();
  rdialog->onZoomFitClicked();
}

void
agaveGui::saveJson()
{
  QFileDialog::Options options;
#ifdef __linux__
  options |= QFileDialog::DontUseNativeDialog;
#endif
  QString file = QFileDialog::getSaveFileName(this, tr("Save JSON"), QString(), tr("JSON (*.json)"), nullptr, options);
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
agaveGui::onImageLoaded(std::shared_ptr<ImageXYZC> image,
                        const LoadSpec& loadSpec,
                        const VolumeDimensions& dims,
                        const ViewerState* vs)
{
  m_loadSpec = loadSpec;

  if (vs) {
    // make sure that ViewerState is consistent with loaded file
    if (dims.sizeT - 1 != vs->m_maxTime) {
      LOG_ERROR << "Mismatch in number of frames: expected " << (vs->m_maxTime + 1) << " and found " << (dims.sizeT)
                << " in the loaded file.";
    }
    if (0 != vs->m_minTime) {
      LOG_ERROR << "Min timline time is not zero.";
    }
  }
  m_currentScene = loadSpec.scene;
  m_appScene.m_timeLine.setRange(0, dims.sizeT - 1);

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
  // everything after the last / (or \ ???) is the filename.

  std::string filename = loadSpec.getFilename();
  m_tabs->setTabText(0, QString::fromStdString(filename));

  m_appearanceDockWidget->onNewImage(&m_appScene);
  m_timelinedock->onNewImage(&m_appScene, loadSpec);

  // set up status view with some stats.
  std::shared_ptr<CStatus> s = m_glView->getStatus();
  // set up the m_statisticsDockWidget as a CStatus  IStatusObserver
  m_statisticsDockWidget->setStatus(s);
  s->onNewImage(filename, &m_appScene);

  m_currentFilePath = loadSpec.filepath;
  agaveGui::prependToRecentFiles(QString::fromStdString(loadSpec.filepath));
  writeRecentDirectory(QString::fromStdString(loadSpec.filepath));
}

bool
agaveGui::open(const std::string& file, const ViewerState* vs)
{
  LoadSpec loadSpec;
  VolumeDimensions dims;

  int sceneToLoad = vs ? vs->m_currentScene : 0;
  int timeToLoad = vs ? vs->m_currentTime : 0;

  if (file.find("http") == 0 || file.find("zarr") != std::string::npos) {
    // read some metadata from the cloud and present the next dialog
    // if successful

    std::vector<MultiscaleDims> multiscaledims;
    bool haveDims = FileReader::loadMultiscaleDims(file, sceneToLoad, multiscaledims);
    if (!haveDims) {
      LOG_DEBUG << "Failed to load dims from url.";
      showOpenFailedMessageBox(QString::fromStdString(file));
      return false;
    }

    // TODO update with sceneToLoad and timeToLoad?
    LoadDialog* loadDialog = new LoadDialog(file, multiscaledims, this);
    if (loadDialog->exec() == QDialog::Accepted) {
      LOG_DEBUG << "OK to load from url.";
      loadSpec = loadDialog->getLoadSpec();
      dims = multiscaledims[loadDialog->getMultiscaleLevelIndex()].getVolumeDimensions();
    } else {
      return false;
    }
  }

  else {
    QFileInfo info(QString::fromStdString(file));
    if (info.exists()) {
      LOG_DEBUG << "Attempting to open " << file;

      // check number of scenes in file.
      int numScenes = FileReader::loadNumScenes(file);
      LOG_INFO << "Found " << numScenes << " scene(s)";
      // if current scene is out of range or if there is not currently a scene selected
      bool needSelectScene = (numScenes > 1) && ((sceneToLoad >= numScenes) || (!vs));
      if (needSelectScene) {
        QStringList items;
        for (int i = 0; i < numScenes; ++i) {
          items.append(QString::number(i));
        }
        bool ok = false;
        QString text = QInputDialog::getItem(this, tr("Select scene"), tr("Scene"), items, m_currentScene, false, &ok);
        if (ok && !text.isEmpty()) {
          sceneToLoad = text.toInt();
        } else {
          LOG_DEBUG << "Canceled scene selection.";
          return false;
        }
      }

      loadSpec.filepath = file;
      loadSpec.scene = sceneToLoad;
      loadSpec.time = timeToLoad;

      dims = FileReader::loadFileDimensions(file, sceneToLoad);
    }
  }

  QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
  std::shared_ptr<ImageXYZC> image = FileReader::loadFromFile(loadSpec);
  QApplication::restoreOverrideCursor();
  if (!image) {
    LOG_DEBUG << "Failed to open " << file;
    showOpenFailedMessageBox(QString::fromStdString(file));
    return false;
  }
  onImageLoaded(image, loadSpec, dims, vs);
  return true;
}

void
agaveGui::openMeshDialog()
{
  QFileDialog::Options options = QFileDialog::DontResolveSymlinks;
#ifdef __linux__
  options |= QFileDialog::DontUseNativeDialog;
#endif
  QString file = QFileDialog::getOpenFileName(this, tr("Open Mesh"), QString(), QString(), 0, options);

  if (!file.isEmpty())
    openMesh(file);
}

void
agaveGui::openMesh(const QString& file)
{
  if (m_appScene.m_volume) {
    return;
  }
}

void
agaveGui::viewFocusChanged(GLView3D* newGlView)
{
  if (m_glView == newGlView)
    return;

  m_viewResetAction->setEnabled(false);

  bool enable(newGlView != 0);

  m_viewResetAction->setEnabled(enable);

  m_glView = newGlView;
}

void
agaveGui::tabChanged(int index)
{
  GLView3D* current = 0;
  if (index >= 0) {
    QWidget* w = m_tabs->currentWidget();
    if (w) {
      current = static_cast<GLView3D*>(w);
    }
  }
  viewFocusChanged(current);
}

void
agaveGui::quit()
{
  close();
}

void
agaveGui::view_reset()
{
  m_glView->initCameraFromImage(&m_appScene);
}

void
agaveGui::view_toggleProjection()
{
  m_glView->toggleCameraProjection();
}

void
agaveGui::setRecentFilesVisible(bool visible)
{
  m_recentFileSubMenuAct->setVisible(visible);
  m_recentFileSeparator->setVisible(visible);
}

static inline QString
recentDirectoryKey()
{
  return QStringLiteral("recentDirectory");
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

void
agaveGui::writeRecentDirectory(const QString& directory)
{
  QSettings settings(QCoreApplication::organizationName(), QCoreApplication::applicationName());
  settings.setValue(recentDirectoryKey(), directory);
}

QString
agaveGui::readRecentDirectory()
{
  QSettings settings(QCoreApplication::organizationName(), QCoreApplication::applicationName());
  QString result = settings.value(recentDirectoryKey()).toString();
  return result;
}

bool
agaveGui::hasRecentFiles()
{
  QSettings settings(QCoreApplication::organizationName(), QCoreApplication::applicationName());
  const int count = settings.beginReadArray(recentFilesKey());
  settings.endArray();
  return count > 0;
}

void
agaveGui::prependToRecentFiles(const QString& fileName)
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
agaveGui::updateRecentFileActions()
{
  QSettings settings(QCoreApplication::organizationName(), QCoreApplication::applicationName());

  const QStringList recentFiles = readRecentFiles(settings);
  const int count = qMin(int(MaxRecentFiles), recentFiles.size());
  int i = 0;
  for (; i < count; ++i) {
    const QString fileName = agaveGui::strippedName(recentFiles.at(i));
    m_recentFileActs[i]->setText(tr("&%1 %2").arg(i + 1).arg(fileName));
    m_recentFileActs[i]->setData(recentFiles.at(i));
    m_recentFileActs[i]->setVisible(true);
  }
  for (; i < MaxRecentFiles; ++i)
    m_recentFileActs[i]->setVisible(false);
}

void
agaveGui::showOpenFailedMessageBox(QString path)
{
  QMessageBox msgBox;
  msgBox.setIcon(QMessageBox::Warning);
  msgBox.setWindowTitle(tr("Error opening file"));
  msgBox.setText(tr("Failed to open ") + path);
  msgBox.setInformativeText(tr("Check logfile.log for more detailed error information."));
  msgBox.exec();
}

void
agaveGui::openRecentFile()
{
  if (const QAction* action = qobject_cast<const QAction*>(sender())) {
    QString path = action->data().toString();
    if (path.endsWith(".obj")) {
      // assume that .obj is mesh
      openMesh(path);
    } else {
      // assumption of ome.tif
      if (!open(path.toStdString())) {
        showOpenFailedMessageBox(path);
      }
    }
  }
}

QString
agaveGui::strippedName(const QString& fullFileName)
{
  if (fullFileName.startsWith("http")) {
    return fullFileName;
  } else {
    return QFileInfo(fullFileName).fileName();
  }
}

void
agaveGui::savePython()
{
  QFileDialog::Options options;
#ifdef __linux__
  options |= QFileDialog::DontUseNativeDialog;
#endif
  QString file = QFileDialog::getSaveFileName(this, tr("Save Python"), QString(), tr("py (*.py)"), nullptr, options);
  if (!file.isEmpty()) {
    QFile saveFile(file);
    if (!saveFile.open(QIODevice::WriteOnly)) {
      qWarning("Couldn't open save file.");
      return;
    }
    ViewerState st = appToViewerState();
    QString doc = st.stateToPythonScript();
    saveFile.write(doc.toUtf8());
  }
}

void
agaveGui::viewerStateToApp(const ViewerState& v)
{
  // ASSUME THAT IMAGE IS LOADED AND APPSCENE INITIALIZED

  // position camera
  m_glView->fromViewerState(v);

  m_appScene.m_roi.SetMinP(glm::vec3(v.m_roiXmin, v.m_roiYmin, v.m_roiZmin));
  m_appScene.m_roi.SetMaxP(glm::vec3(v.m_roiXmax, v.m_roiYmax, v.m_roiZmax));

  m_appScene.m_timeLine.setRange(v.m_minTime, v.m_maxTime);
  m_appScene.m_timeLine.setCurrentTime(v.m_currentTime);

  m_currentScene = v.m_currentScene;

  m_appScene.m_volume->setPhysicalSize(v.m_scaleX, v.m_scaleY, v.m_scaleZ);

  m_appScene.m_material.m_backgroundColor[0] = v.m_backgroundColor.x;
  m_appScene.m_material.m_backgroundColor[1] = v.m_backgroundColor.y;
  m_appScene.m_material.m_backgroundColor[2] = v.m_backgroundColor.z;

  m_appScene.m_material.m_boundingBoxColor[0] = v.m_boundingBoxColor.x;
  m_appScene.m_material.m_boundingBoxColor[1] = v.m_boundingBoxColor.y;
  m_appScene.m_material.m_boundingBoxColor[2] = v.m_boundingBoxColor.z;

  m_appScene.m_material.m_showBoundingBox = v.m_showBoundingBox;

  m_renderSettings.m_RenderSettings.m_DensityScale = v.m_densityScale;
  m_renderSettings.m_RenderSettings.m_StepSizeFactor = v.m_primaryStepSize;
  m_renderSettings.m_RenderSettings.m_StepSizeFactorShadow = v.m_secondaryStepSize;
  m_renderSettings.m_RenderSettings.m_GradientFactor = v.m_gradientFactor;

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
    m_appScene.m_material.m_opacity[i] = ch.m_opacity;

    m_appScene.m_material.m_gradientData[i].m_activeMode = LutParams::g_PermIdToGradientMode[ch.m_lutParams.m_mode];
    m_appScene.m_material.m_gradientData[i].m_window = ch.m_lutParams.m_window;
    m_appScene.m_material.m_gradientData[i].m_level = ch.m_lutParams.m_level;
    m_appScene.m_material.m_gradientData[i].m_pctLow = ch.m_lutParams.m_pctLow;
    m_appScene.m_material.m_gradientData[i].m_pctHigh = ch.m_lutParams.m_pctHigh;
    m_appScene.m_material.m_gradientData[i].m_isovalue = ch.m_lutParams.m_isovalue;
    m_appScene.m_material.m_gradientData[i].m_isorange = ch.m_lutParams.m_isorange;
    m_appScene.m_material.m_gradientData[i].m_customControlPoints = ch.m_lutParams.m_customControlPoints;
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

  // capture settings
  m_captureSettings.width = v.m_captureState.mWidth;
  m_captureSettings.height = v.m_captureState.mHeight;
  m_captureSettings.samples = v.m_captureState.mSamples;
  m_captureSettings.duration = v.m_captureState.mDuration;
  m_captureSettings.durationType = (eRenderDurationType)v.m_captureState.mDurationType;
  m_captureSettings.startTime = v.m_captureState.mStartTime;
  m_captureSettings.endTime = v.m_captureState.mEndTime;
  m_captureSettings.outputDir = v.m_captureState.mOutputDir;
  m_captureSettings.filenamePrefix = v.m_captureState.mFilenamePrefix;

  m_renderSettings.m_DirtyFlags.SetFlag(CameraDirty);
  m_renderSettings.m_DirtyFlags.SetFlag(LightsDirty);
  m_renderSettings.m_DirtyFlags.SetFlag(RenderParamsDirty);
  m_renderSettings.m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

ViewerState
agaveGui::appToViewerState()
{
  ViewerState v;
  v.m_volumeImageFile = m_currentFilePath;

  if (m_appScene.m_volume) {
    v.m_scaleX = m_appScene.m_volume->physicalSizeX();
    v.m_scaleY = m_appScene.m_volume->physicalSizeY();
    v.m_scaleZ = m_appScene.m_volume->physicalSizeZ();
  }

  v.m_backgroundColor = glm::vec3(m_appScene.m_material.m_backgroundColor[0],
                                  m_appScene.m_material.m_backgroundColor[1],
                                  m_appScene.m_material.m_backgroundColor[2]);

  v.m_boundingBoxColor = glm::vec3(m_appScene.m_material.m_boundingBoxColor[0],
                                   m_appScene.m_material.m_boundingBoxColor[1],
                                   m_appScene.m_material.m_boundingBoxColor[2]);
  v.m_showBoundingBox = m_appScene.m_material.m_showBoundingBox;

  v.m_resolutionX = m_glView->size().width();
  v.m_resolutionY = m_glView->size().height();
  v.m_renderIterations = m_renderSettings.GetNoIterations();

  v.m_minTime = m_appScene.m_timeLine.minTime();
  v.m_maxTime = m_appScene.m_timeLine.maxTime();
  v.m_currentTime = m_appScene.m_timeLine.currentTime();

  v.m_currentScene = m_currentScene;

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

  v.m_projection = m_glView->getCamera().m_Projection == PERSPECTIVE ? ViewerState::Projection::PERSPECTIVE
                                                                     : ViewerState::Projection::ORTHOGRAPHIC;
  v.m_orthoScale = m_glView->getCamera().m_OrthoScale;
  v.m_fov = m_qcamera.GetProjection().GetFieldOfView();

  v.m_exposure = m_qcamera.GetFilm().GetExposure();
  v.m_apertureSize = m_qcamera.GetAperture().GetSize();
  v.m_focalDistance = m_qcamera.GetFocus().GetFocalDistance();
  v.m_densityScale = m_renderSettings.m_RenderSettings.m_DensityScale;
  v.m_gradientFactor = m_renderSettings.m_RenderSettings.m_GradientFactor;

  v.m_primaryStepSize = m_renderSettings.m_RenderSettings.m_StepSizeFactor;
  v.m_secondaryStepSize = m_renderSettings.m_RenderSettings.m_StepSizeFactorShadow;

  if (m_appScene.m_volume) {
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
      ch.m_opacity = m_appScene.m_material.m_opacity[i];

      ch.m_lutParams.m_mode = LutParams::g_GradientModeToPermId[m_appScene.m_material.m_gradientData[i].m_activeMode];
      ch.m_lutParams.m_window = m_appScene.m_material.m_gradientData[i].m_window;
      ch.m_lutParams.m_level = m_appScene.m_material.m_gradientData[i].m_level;
      ch.m_lutParams.m_pctLow = m_appScene.m_material.m_gradientData[i].m_pctLow;
      ch.m_lutParams.m_pctHigh = m_appScene.m_material.m_gradientData[i].m_pctHigh;
      ch.m_lutParams.m_isovalue = m_appScene.m_material.m_gradientData[i].m_isovalue;
      ch.m_lutParams.m_isorange = m_appScene.m_material.m_gradientData[i].m_isorange;
      ch.m_lutParams.m_customControlPoints = m_appScene.m_material.m_gradientData[i].m_customControlPoints;

      v.m_channels.push_back(ch);
    }
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

  // capture settings
  v.m_captureState.mWidth = m_captureSettings.width;
  v.m_captureState.mHeight = m_captureSettings.height;
  v.m_captureState.mSamples = m_captureSettings.samples;
  v.m_captureState.mDuration = m_captureSettings.duration;
  v.m_captureState.mDurationType = m_captureSettings.durationType;
  v.m_captureState.mStartTime = m_captureSettings.startTime;
  v.m_captureState.mEndTime = m_captureSettings.endTime;
  v.m_captureState.mOutputDir = m_captureSettings.outputDir;
  v.m_captureState.mFilenamePrefix = m_captureSettings.filenamePrefix;

  return v;
}
