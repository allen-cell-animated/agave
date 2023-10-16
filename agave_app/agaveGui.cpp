// Include first to avoid clash with Windows headers pulled in via
// QtCore/qt_windows.h; they define VOID and HALFTONE which clash with
// the TIFF enums.
#include <memory>

#include "agaveGui.h"

#include "renderlib/AppScene.h"
#include "renderlib/ImageXYZC.h"
#include "renderlib/Logging.h"
#include "renderlib/Status.h"
#include "renderlib/VolumeDimensions.h"
#include "renderlib/io/FileReader.h"
#include "renderlib/version.hpp"

#include "AppearanceDockWidget.h"
#include "CameraDockWidget.h"
#include "Serialize.h"
#include "StatisticsDockWidget.h"
#include "TimelineDockWidget.h"
#include "ViewerState.h"
#include "loadDialog.h"
#include "renderDialog.h"

#include <QAction>
#include <QElapsedTimer>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QSettings>
#include <QToolBar>

#include <filesystem>

agaveGui::agaveGui(QWidget* parent)
  : QMainWindow(parent)
{
  setStyleSheet("QToolTip{padding:3px;}");
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
  m_glView = new WgpuCanvas(&m_qcamera, &m_qrendersettings, &m_renderSettings, this);
  QObject::connect(m_glView, SIGNAL(ChangedRenderer()), this, SLOT(OnUpdateRenderer()));

  m_glView->setObjectName("glcontainer");
  // We need a minimum size or else the size defaults to zero.
  m_glView->setMinimumSize(256, 512);
  m_tabs->addTab(m_glView, "None");

  QString windowTitle =
    QApplication::instance()->applicationName() + " " + QApplication::instance()->applicationVersion();
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
  m_openAction = new QAction(tr("&Open file (.tiff, ...)"), this);
  m_openAction->setShortcuts(QKeySequence::Open);
  m_openAction->setStatusTip(tr("Open an existing volume file"));
  connect(m_openAction, SIGNAL(triggered()), this, SLOT(open()));

  m_openUrlAction = new QAction(tr("&Open from URL"), this);
  m_openUrlAction->setShortcuts(QKeySequence::Open);
  m_openUrlAction->setStatusTip(tr("Open an existing volume in the cloud"));
  connect(m_openUrlAction, SIGNAL(triggered()), this, SLOT(openUrl()));

  m_openDirectoryAction = new QAction(tr("&Open directory (.zarr)"), this);
  m_openDirectoryAction->setShortcuts(QKeySequence::Open);
  m_openDirectoryAction->setStatusTip(tr("Open an existing volume from local directory"));
  connect(m_openDirectoryAction, SIGNAL(triggered()), this, SLOT(openDirectory()));

  m_openJsonAction = new QAction(tr("Open JSON..."), this);
  m_openJsonAction->setStatusTip(tr("Open an existing JSON settings file"));
  connect(m_openJsonAction, SIGNAL(triggered()), this, SLOT(openJson()));

  m_quitAction = new QAction(tr("&Quit"), this);
  m_quitAction->setShortcuts(QKeySequence::Quit);
  m_quitAction->setStatusTip(tr("Quit the application"));
  connect(m_quitAction, SIGNAL(triggered()), this, SLOT(quit()));

  m_viewResetAction = new QAction(tr("&Reset"), this);
  m_viewResetAction->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_R));
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
  m_fileMenu->addAction(m_openDirectoryAction);
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
  m_ui.mainToolBar->addAction(m_openDirectoryAction);
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
  m_timelinedock->setVisible(false); // hide by default

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
void
agaveGui::openDirectory()
{
  QString dir = readRecentDirectory();

  QFileDialog::Options options = QFileDialog::DontResolveSymlinks | QFileDialog::ShowDirsOnly;
#ifdef __linux__
  options |= QFileDialog::DontUseNativeDialog;
#endif
  QString file = QFileDialog::getExistingDirectory(this, tr("Open Volume"), dir, options);

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
  QInputDialog dlg(this);
  dlg.setInputMode(QInputDialog::TextInput);
  dlg.setLabelText(tr("Enter URL here:"));
  dlg.setWindowTitle(tr("Open from URL"));
  dlg.setTextValue("");
  dlg.setInputMethodHints(Qt::ImhUrlCharactersOnly);
  dlg.resize(400, dlg.sizeHint().height());
  bool ok = dlg.exec();
  QString text = dlg.textValue();
  if (ok && !text.isEmpty()) {
    urlToLoad = text.toStdString();
  } else {
    LOG_DEBUG << "Canceled load Zarr from url.";
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

    // get the bytes into a ViewerState object.
    try {
      std::string saveDataString = saveData.toStdString();
      nlohmann::json j = nlohmann::json::parse(saveDataString);
      Serialize::ViewerState s;
      s = stateFromJson(j);
      if (!s.datasets.empty()) {
        if (!open(s.datasets[0].url, &s)) {
          showOpenFailedMessageBox(file);
        }
      }
    } catch (std::exception& e) {
      LOG_ERROR << "Failed to load from JSON: " << file.toStdString();
      LOG_ERROR << e.what();
      return;
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
  // TODO keep this loadspec time in sync with the timeline and the render dialog's time
  m_loadSpec.time = m_appScene.m_timeLine.currentTime();

  // if we are disabling the 3d view then might consider just making this modal
  m_glView->pauseRenderLoop();
  // QImage im = m_glView->captureQimage();
  // QImage* imcopy = new QImage(im);
  m_glView->doneCurrent();
  m_glView->setEnabled(false);
  m_glView->setUpdatesEnabled(false);
  // extract Renderer from GLView3D to hand to RenderDialog
  IRenderWindow* renderer = m_glView->borrowRenderer();
  if (m_captureSettings.width == 0 && m_captureSettings.height == 0) {
    m_captureSettings.width = m_glView->width();
    m_captureSettings.height = m_glView->height();
  }

  // TODO should we reuse the last settings for capture start and end time?
  // currently every time you enter the render window we are putting things to the current time.
  m_captureSettings.startTime = m_appScene.m_timeLine.currentTime();
  m_captureSettings.endTime = m_appScene.m_timeLine.currentTime();

  // copy of camera
  CCamera camera = m_glView->getCamera();
  RenderDialog* rdialog = new RenderDialog(renderer,
                                           m_renderSettings,
                                           m_appScene,
                                           camera,
                                           m_glView->context(),
                                           m_loadSpec,
                                           &m_captureSettings,
                                           m_glView->width(),
                                           m_glView->height(),
                                           this);
  rdialog->resize(geometry().width(), m_tabs->height());
  rdialog->move(geometry().x(), geometry().y());
  connect(rdialog, &QDialog::finished, this, [this, &rdialog](int result) {
    // get renderer from RenderDialog and hand it back to GLView3D
    LOG_DEBUG << "RenderDialog finished with result " << result;
    m_glView->setEnabled(true);
    m_glView->resizeGL(m_glView->width(), m_glView->height());
    m_glView->setUpdatesEnabled(true);
    m_glView->restartRenderLoop();
    // refresh timeline to current time
    m_timelinedock->setTime(m_appScene.m_timeLine.currentTime());
  });

  // rdialog->setImage(imcopy);
  // delete imcopy;
  rdialog->setModal(true);
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

    QFile saveFile(file);
    if (!saveFile.open(QIODevice::WriteOnly)) {
      qWarning("Couldn't open save file.");
      return;
    }
    Serialize::ViewerState st = appToViewerState();
    nlohmann::json doc = st;
    std::string str = doc.dump();
    saveFile.write(str.c_str()); // QString::fromStdString(str));
  }
}

void
agaveGui::onImageLoaded(std::shared_ptr<ImageXYZC> image,
                        const LoadSpec& loadSpec,
                        uint32_t sizeT,
                        const Serialize::ViewerState* vs,
                        std::shared_ptr<IFileReader> reader)
{
  m_loadSpec = loadSpec;

  if (vs) {
    // make sure that ViewerState is consistent with loaded file
    if (sizeT - 1 != vs->timeline.maxTime) {
      LOG_ERROR << "Mismatch in number of frames: expected " << (vs->timeline.maxTime + 1) << " and found " << (sizeT)
                << " in the loaded file.";
    }
    if (0 != vs->timeline.minTime) {
      LOG_ERROR << "Min timline time is not zero.";
    }
  }
  m_currentScene = loadSpec.scene;
  m_appScene.m_timeLine.setRange(0, sizeT - 1);
  m_appScene.m_timeLine.setCurrentTime(loadSpec.time);

  // Show timeline widget if the loaded image has multiple frames.
  m_timelinedock->setVisible(sizeT > 1);

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
  m_timelinedock->onNewImage(&m_appScene, loadSpec, reader);

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
agaveGui::open(const std::string& file, const Serialize::ViewerState* vs)
{
  LoadSpec loadSpec;
  VolumeDimensions dims;

  int sceneToLoad = 0;
  int timeToLoad = 0;
  if (vs) {
    loadSpec = stateToLoadSpec(*vs);
    sceneToLoad = loadSpec.scene;
    timeToLoad = loadSpec.time;
  }

  std::shared_ptr<IFileReader> reader(FileReader::getReader(file));
  if (!reader) {
    QMessageBox b(QMessageBox::Warning,
                  "Error",
                  "Could not determine filetype of \"" + QString::fromStdString(file) +
                    "\".  Make sure you supply a valid URL or filepath to a file supported by AGAVE.",
                  QMessageBox::Ok,
                  this);
    b.exec();
    LOG_ERROR << "Could not find a reader for file " << file;
    return false;
  }

  // read some metadata from the cloud and present the next dialog
  // if successful
  int numScenes = reader->loadNumScenes(file);
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

  std::vector<MultiscaleDims> multiscaledims;
  multiscaledims = reader->loadMultiscaleDims(file, sceneToLoad);
  if (multiscaledims.empty()) {
    LOG_DEBUG << "Failed to load dims for image.";
    showOpenFailedMessageBox(QString::fromStdString(file));
    return false;
  }

  if (!vs) {

    LoadDialog* loadDialog = new LoadDialog(file, multiscaledims, sceneToLoad, this);
    if (loadDialog->exec() == QDialog::Accepted) {
      loadSpec = loadDialog->getLoadSpec();
      dims = multiscaledims[loadDialog->getMultiscaleLevelIndex()].getVolumeDimensions();
    } else {
      LOG_INFO << "Canceled load dialog.";
      return true;
    }
    delete loadDialog;

  } else {
    // we called stateToLoadSpec above
    // now it is only necessary to get the dims for onImageLoaded...
    // huge assumption that level 0 has sizeT at least as large as the others?
    dims = multiscaledims[0].getVolumeDimensions();
  }

  // TODO make this part async and chunked so that it can be interrupted
  // and won't block during long loading times.
  // We can update the render and gui progressively as chunks are loaded.
  // Also, this would allow renders to be cancelled during loading.
  QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
  std::shared_ptr<ImageXYZC> image = reader->loadFromFile(loadSpec);
  QApplication::restoreOverrideCursor();
  if (!image) {
    LOG_DEBUG << "Failed to open " << file;
    showOpenFailedMessageBox(QString::fromStdString(file));
    return false;
  }
  onImageLoaded(image, loadSpec, dims.sizeT, vs, reader);
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
agaveGui::viewFocusChanged(WgpuCanvas* newGlView)
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
  WgpuCanvas* current = 0;
  if (index >= 0) {
    QWidget* w = m_tabs->currentWidget();
    if (w) {
      current = static_cast<WgpuCanvas*>(w);
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
    Serialize::ViewerState st = appToViewerState();
    QString doc = stateToPythonScript(st);
    saveFile.write(doc.toUtf8());
  }
}

void
agaveGui::viewerStateToApp(const Serialize::ViewerState& v)
{
  // ASSUME THAT IMAGE IS LOADED AND APPSCENE INITIALIZED

  // position camera
  m_glView->fromViewerState(v);

  m_appScene.m_roi.SetMinP(glm::vec3(v.clipRegion[0][0], v.clipRegion[1][0], v.clipRegion[2][0]));
  m_appScene.m_roi.SetMaxP(glm::vec3(v.clipRegion[0][1], v.clipRegion[1][1], v.clipRegion[2][1]));

  m_appScene.m_timeLine.setRange(v.timeline.minTime, v.timeline.maxTime);
  m_appScene.m_timeLine.setCurrentTime(v.timeline.currentTime);

  m_currentScene = v.datasets[0].scene;

  m_appScene.m_volume->setPhysicalSize(v.scale[0], v.scale[1], v.scale[2]);

  m_appScene.m_material.m_backgroundColor[0] = v.backgroundColor[0];
  m_appScene.m_material.m_backgroundColor[1] = v.backgroundColor[1];
  m_appScene.m_material.m_backgroundColor[2] = v.backgroundColor[2];

  m_appScene.m_material.m_boundingBoxColor[0] = v.boundingBoxColor[0];
  m_appScene.m_material.m_boundingBoxColor[1] = v.boundingBoxColor[1];
  m_appScene.m_material.m_boundingBoxColor[2] = v.boundingBoxColor[2];

  m_appScene.m_material.m_showBoundingBox = v.showBoundingBox;

  m_renderSettings.m_RenderSettings.m_DensityScale = v.density;
  m_renderSettings.m_RenderSettings.m_StepSizeFactor = v.pathTracer.primaryStepSize;
  m_renderSettings.m_RenderSettings.m_StepSizeFactorShadow = v.pathTracer.secondaryStepSize;
  // m_renderSettings.m_RenderSettings.m_GradientFactor = v.m_gradientFactor;

  // channels
  for (uint32_t i = 0; i < m_appScene.m_volume->sizeC(); ++i) {
    Serialize::ChannelSettings_V1 ch = v.channels[i];
    m_appScene.m_material.m_enabled[i] = ch.enabled;

    m_appScene.m_material.m_diffuse[i * 3] = ch.diffuseColor[0];
    m_appScene.m_material.m_diffuse[i * 3 + 1] = ch.diffuseColor[1];
    m_appScene.m_material.m_diffuse[i * 3 + 2] = ch.diffuseColor[2];

    m_appScene.m_material.m_specular[i * 3] = ch.specularColor[0];
    m_appScene.m_material.m_specular[i * 3 + 1] = ch.specularColor[1];
    m_appScene.m_material.m_specular[i * 3 + 2] = ch.specularColor[2];

    m_appScene.m_material.m_emissive[i * 3] = ch.emissiveColor[0];
    m_appScene.m_material.m_emissive[i * 3 + 1] = ch.emissiveColor[1];
    m_appScene.m_material.m_emissive[i * 3 + 2] = ch.emissiveColor[2];

    m_appScene.m_material.m_roughness[i] = ch.glossiness;
    m_appScene.m_material.m_opacity[i] = ch.opacity;

    m_appScene.m_material.m_gradientData[i] = stateToGradientData(v, i);
  }

  // lights
  m_appScene.m_lighting.m_Lights[0] = stateToLight(v, 0);
  m_appScene.m_lighting.m_Lights[1] = stateToLight(v, 1);

  // capture settings
  m_captureSettings.width = v.capture.width;
  m_captureSettings.height = v.capture.height;
  m_captureSettings.samples = v.capture.samples;
  m_captureSettings.duration = v.capture.seconds;
  // TODO proper lookup for permid
  m_captureSettings.durationType = (eRenderDurationType)v.capture.durationType;
  m_captureSettings.startTime = v.capture.startTime;
  m_captureSettings.endTime = v.capture.endTime;
  m_captureSettings.outputDir = v.capture.outputDirectory;
  m_captureSettings.filenamePrefix = v.capture.filenamePrefix;

  m_renderSettings.m_DirtyFlags.SetFlag(CameraDirty);
  m_renderSettings.m_DirtyFlags.SetFlag(LightsDirty);
  m_renderSettings.m_DirtyFlags.SetFlag(RenderParamsDirty);
  m_renderSettings.m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

Serialize::ViewerState
agaveGui::appToViewerState()
{
  Serialize::ViewerState v;
  v.version = { AICS_VERSION_MAJOR, AICS_VERSION_MINOR, AICS_VERSION_PATCH };

  v.datasets.push_back(fromLoadSpec(m_loadSpec));

  if (m_appScene.m_volume) {
    v.scale[0] = m_appScene.m_volume->physicalSizeX();
    v.scale[1] = m_appScene.m_volume->physicalSizeY();
    v.scale[2] = m_appScene.m_volume->physicalSizeZ();
  }

  v.backgroundColor = { m_appScene.m_material.m_backgroundColor[0],
                        m_appScene.m_material.m_backgroundColor[1],
                        m_appScene.m_material.m_backgroundColor[2] };

  v.boundingBoxColor = { m_appScene.m_material.m_boundingBoxColor[0],
                         m_appScene.m_material.m_boundingBoxColor[1],
                         m_appScene.m_material.m_boundingBoxColor[2] };
  v.showBoundingBox = m_appScene.m_material.m_showBoundingBox;

  v.capture.samples = m_renderSettings.GetNoIterations();

  v.timeline.minTime = m_appScene.m_timeLine.minTime();
  v.timeline.maxTime = m_appScene.m_timeLine.maxTime();
  v.timeline.currentTime = m_appScene.m_timeLine.currentTime();

  v.clipRegion[0][1] = m_appScene.m_roi.GetMaxP().x;
  v.clipRegion[1][1] = m_appScene.m_roi.GetMaxP().y;
  v.clipRegion[2][1] = m_appScene.m_roi.GetMaxP().z;
  v.clipRegion[0][0] = m_appScene.m_roi.GetMinP().x;
  v.clipRegion[1][0] = m_appScene.m_roi.GetMinP().y;
  v.clipRegion[2][0] = m_appScene.m_roi.GetMinP().z;

  v.camera.eye[0] = m_glView->getCamera().m_From.x;
  v.camera.eye[1] = m_glView->getCamera().m_From.y;
  v.camera.eye[2] = m_glView->getCamera().m_From.z;

  v.camera.target[0] = m_glView->getCamera().m_Target.x;
  v.camera.target[1] = m_glView->getCamera().m_Target.y;
  v.camera.target[2] = m_glView->getCamera().m_Target.z;

  v.camera.up[0] = m_glView->getCamera().m_Up.x;
  v.camera.up[1] = m_glView->getCamera().m_Up.y;
  v.camera.up[2] = m_glView->getCamera().m_Up.z;

  v.camera.projection = m_glView->getCamera().m_Projection == PERSPECTIVE ? Serialize::Projection_PID::PERSPECTIVE
                                                                          : Serialize::Projection_PID::ORTHOGRAPHIC;
  v.camera.orthoScale = m_glView->getCamera().m_OrthoScale;
  v.camera.fovY = m_qcamera.GetProjection().GetFieldOfView();

  v.camera.exposure = m_qcamera.GetFilm().GetExposure();
  v.camera.aperture = m_qcamera.GetAperture().GetSize();
  v.camera.focalDistance = m_qcamera.GetFocus().GetFocalDistance();
  v.density = m_renderSettings.m_RenderSettings.m_DensityScale;
  // v.m_gradientFactor = m_renderSettings.m_RenderSettings.m_GradientFactor;

  v.rendererType = m_qrendersettings.GetRendererType() == 0 ? Serialize::RendererType_PID::RAYMARCH
                                                            : Serialize::RendererType_PID::PATHTRACE;

  v.pathTracer.primaryStepSize = m_renderSettings.m_RenderSettings.m_StepSizeFactor;
  v.pathTracer.secondaryStepSize = m_renderSettings.m_RenderSettings.m_StepSizeFactorShadow;

  if (m_appScene.m_volume) {
    for (uint32_t i = 0; i < m_appScene.m_volume->sizeC(); ++i) {
      Serialize::ChannelSettings_V1 ch;
      ch.enabled = m_appScene.m_material.m_enabled[i];
      ch.diffuseColor = { m_appScene.m_material.m_diffuse[i * 3],
                          m_appScene.m_material.m_diffuse[i * 3 + 1],
                          m_appScene.m_material.m_diffuse[i * 3 + 2] };
      ch.specularColor = { m_appScene.m_material.m_specular[i * 3],
                           m_appScene.m_material.m_specular[i * 3 + 1],
                           m_appScene.m_material.m_specular[i * 3 + 2] };
      ch.emissiveColor = { m_appScene.m_material.m_emissive[i * 3],
                           m_appScene.m_material.m_emissive[i * 3 + 1],
                           m_appScene.m_material.m_emissive[i * 3 + 2] };
      ch.glossiness = m_appScene.m_material.m_roughness[i];
      ch.opacity = m_appScene.m_material.m_opacity[i];

      ch.lutParams = fromGradientData(m_appScene.m_material.m_gradientData[i]);

      v.channels.push_back(ch);
    }
  }

  // lighting
  Light& lt = m_appScene.m_lighting.m_Lights[0];
  Serialize::LightSettings_V1 l = fromLight(lt);
  v.lights.push_back(l);

  Light& lt1 = m_appScene.m_lighting.m_Lights[1];
  Serialize::LightSettings_V1 l1 = fromLight(lt1);
  v.lights.push_back(l1);

  // capture settings

  v.capture = fromCaptureSettings(m_captureSettings, m_glView->width(), m_glView->height());

  return v;
}
