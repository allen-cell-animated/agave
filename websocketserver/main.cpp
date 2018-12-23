#include <QApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "mainwindow.h"
#include "renderlib/FileReader.h"
#include "renderlib/Logging.h"
#include "renderlib/renderlib.h"
#include "renderlib/version.h"
#include "streamserver.h"

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
    qDebug() << "No server config file found openable at " << configPath;
    return p;
  }

  QByteArray jsonData = loadFile.readAll();
  QJsonDocument jsonDoc(QJsonDocument::fromJson(jsonData));
  if (jsonDoc.isNull()) {
    qDebug() << "Invalid server config file format. Make sure it is json.";
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
      renderlib::imageAllocGPU_Cuda(img);
    } else {
      qDebug() << "Could not load " << s;
    }
  }
}

int
main(int argc, char* argv[])
{
  // set to true to show windows, or false to run as a console application
  bool gui = false;

  QApplication::setAttribute(Qt::AA_UseDesktopOpenGL, true);
  QApplication a(argc, argv);
  // QDir::setCurrent(QApplication::applicationDirPath() + "/work");

  a.setOrganizationName("AICS");
  a.setOrganizationDomain("allencell.org");
  a.setApplicationName("AICS RENDERSERVER");
  a.setApplicationVersion(AICS_VERSION_STRING);
  LOG_INFO << a.organizationName().toStdString() << " " << a.applicationName().toStdString() << " " << a.applicationVersion().toStdString();

  QCommandLineParser parser;
  parser.setApplicationDescription("Remote rendering service via websockets");
  parser.addHelpOption();
  parser.addVersionOption();
  parser.addPositionalArgument("config",
                               QCoreApplication::translate("main", "Config file path. Default to ./setup.cfg"));

  // Process the actual command line arguments given by the user
  parser.process(a);

  const QStringList args = parser.positionalArguments();

  QString configPath(QStringLiteral("server.cfg"));
  if (args.size() > 0) {
    configPath = args.at(0);
  }
  ServerParams p = readConfig(configPath);

  if (!renderlib::initialize()) {
    renderlib::cleanup();
    return 0;
  }

  StreamServer* server = new StreamServer(p._port, false, 0);

  if (gui) {
    MainWindow* _ = new MainWindow(server);
    _->resize(512, 512);
    _->show();
  }

  qDebug() << "working directory:" << QDir::current();

  // delete logFile;

  // must happen after renderlib init
  preloadFiles(p._preloadList);

  int result = a.exec();
  renderlib::cleanup();
  return result;
}
