#include "agaveGui.h"

#include "mainwindow.h"
#include "renderlib/FileReader.h"
#include "renderlib/Logging.h"
#include "renderlib/renderlib.h"
#include "renderlib/version.h"
#include "streamserver.h"
#include "streamtestclient.h"

#include <QApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

struct ServerParams
{
  int _port;
  QStringList _preloadList;

  // defaults
  ServerParams()
    : _port(1235)
  {
  }
};

ServerParams
readConfig(QString configPath)
{
  // defaults from default ctor
  ServerParams p;

  // try to open server.cfg
  QFile loadFile(configPath);
  if (!loadFile.open(QIODevice::ReadOnly)) {
    LOG_INFO << "No server config file found openable at " << configPath.toStdString() << " . Using defaults.";
    return p;
  }

  QByteArray jsonData = loadFile.readAll();
  QJsonDocument jsonDoc(QJsonDocument::fromJson(jsonData));
  if (jsonDoc.isNull()) {
    LOG_INFO << "Invalid server config file format. Make sure it is JSON.";
    return p;
  }

  QJsonObject json(jsonDoc.object());

  // server config file:
  // {
  //   port: 1235,
  //   preload: ['/path/to/file1', '/path/to/file2', ...]
  // }

  if (json.contains("port") /* && json["port"].isDouble()*/) {
    p._port = json["port"].toInt(p._port);
  }

  if (json.contains("preload") && json["preload"].isArray()) {
    QJsonArray preloadArray = json["preload"].toArray();
    p._preloadList.clear();
    p._preloadList.reserve(preloadArray.size());
    for (int i = 0; i < preloadArray.size(); ++i) {
      QString preloadString = preloadArray[i].toString();
      p._preloadList.append(preloadString);
    }
  }

  return p;
}

void
preloadFiles(QStringList preloadlist)
{
  for (QString s : preloadlist) {
    QFileInfo info(s);
    if (info.exists()) {
      auto img = FileReader::loadFromFile_4D(info.absoluteFilePath().toStdString(), nullptr, true);
      renderlib::imageAllocGPU(img);
    } else {
      LOG_INFO << "Could not load " << s.toStdString();
    }
  }
}

int
main(int argc, char* argv[])
{
  Logging::Init();

  QApplication::setAttribute(Qt::AA_UseDesktopOpenGL);
  QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QApplication a(argc, argv);
  a.setOrganizationName("AICS");
  a.setOrganizationDomain("allencell.org");
  a.setApplicationName("GPU Volume Explorer");
  a.setApplicationVersion(AICS_VERSION_STRING);

  LOG_INFO << a.organizationName().toStdString() << " " << a.applicationName().toStdString() << " "
           << a.applicationVersion().toStdString();

  QCommandLineParser parser;
  parser.setApplicationDescription("Advanced GPU Accelerated Volume Explorer");
  parser.addHelpOption();
  parser.addVersionOption();
  QCommandLineOption serverOption("server",
                                  QCoreApplication::translate("main", "Run as websocket server without GUI."));
  parser.addOption(serverOption);
  QCommandLineOption listDevicesOption(
    "list_devices", QCoreApplication::translate("main", "Log the known EGL devices (only valid in --server mode)."));
  parser.addOption(listDevicesOption);
  QCommandLineOption selectGpuOption(
    "gpu",
    QCoreApplication::translate("main", "Select EGL device by index (only valid in --server mode)."),
    QCoreApplication::translate("main", "gpu"),
    "0");
  parser.addOption(selectGpuOption);
  QCommandLineOption serverConfigOption("config",
                                        QCoreApplication::translate("main", "Path to config file."),
                                        QCoreApplication::translate("main", "config"),
                                        QCoreApplication::translate("main", "setup.cfg"));
  parser.addOption(serverConfigOption);

  // Process the actual command line arguments given by the user
  parser.process(a);

  bool isServer = parser.isSet(serverOption);
  bool listDevices = parser.isSet(listDevicesOption);
  int selectedGpu = parser.value(selectGpuOption).toInt();

  if (!renderlib::initialize(isServer, listDevices, selectedGpu)) {
    renderlib::cleanup();
    return 0;
  }

  int result = 0;

  if (isServer) {
    QString configPath = parser.value(serverConfigOption);
    ServerParams p = readConfig(configPath);

    StreamServer* server = new StreamServer(p._port, false, 0);

    // set to true to show windows, or false to run as a console application
    static const bool gui = false;
    static const bool serverTest = true;
    if (gui) {
      MainWindow* _ = new MainWindow(server);
      _->resize(512, 512);
      _->show();
    } else if (serverTest) {
      StreamTestClient* client = new StreamTestClient(QUrl("ws://localhost:1235"), true);
    }

    LOG_INFO << "Created server at working directory:" << QDir::currentPath().toStdString();

    // delete logFile;

    // must happen after renderlib init
    preloadFiles(p._preloadList);

    result = a.exec();
  } else {
    try {
      agaveGui* w = new agaveGui();
      w->show();
      result = a.exec();

    } catch (...) {
      LOG_ERROR << "Exception caught in main";
    }
  }

  renderlib::cleanup();

  return result;
}
