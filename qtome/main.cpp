#include "qtome.h"

#include "mainwindow.h"
#include "renderlib/FileReader.h"
#include "renderlib/Logging.h"
#include "renderlib/renderlib.h"
#include "renderlib/version.h"
#include "streamserver.h"

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
  {}
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
    LOG_INFO << "Invalid server config file format. Make sure it is json.";
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
      auto img = FileReader::loadOMETiff_4D(info.absoluteFilePath().toStdString(), true);
      renderlib::imageAllocGPU(img);
    } else {
      LOG_INFO << "Could not load " << s.toStdString();
    }
  }
}

int
main(int argc, char* argv[])
{
  LOG_INFO << ("started");
  QApplication::setAttribute(Qt::AA_UseDesktopOpenGL);
  QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QApplication a(argc, argv);
  LOG_INFO << ("created QApplication object");
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
  QCommandLineOption serverOption("server", QCoreApplication::translate("main", "Run as websocket server without GUI"));
  parser.addOption(serverOption);
  QCommandLineOption serverConfigOption("config",
                                        QCoreApplication::translate("main", "Path to config file"),
                                        QCoreApplication::translate("main", "config"),
                                        QCoreApplication::translate("main", "setup.cfg"));
  parser.addOption(serverConfigOption);

  // Process the actual command line arguments given by the user
  parser.process(a);

  if (!renderlib::initialize()) {
    renderlib::cleanup();
    return 0;
  }
  LOG_INFO << ("initialized renderlib");

  bool isServer = parser.isSet(serverOption);
  if (isServer) {
    QString configPath = parser.value(serverConfigOption);
    ServerParams p = readConfig(configPath);

    StreamServer* server = new StreamServer(p._port, false, 0);
    LOG_INFO << ("created server");

    // set to true to show windows, or false to run as a console application
    static const bool gui = false;
    if (gui) {
      MainWindow* _ = new MainWindow(server);
      _->resize(512, 512);
      _->show();
    }

    LOG_INFO << "working directory:" << QDir::currentPath().toStdString();

    // delete logFile;

    // must happen after renderlib init
    preloadFiles(p._preloadList);

  } else {
    qtome* w = new qtome();
    w->show();
  }

  int result = a.exec();

  renderlib::cleanup();

  return result;
}
