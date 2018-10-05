#include <QApplication>
#include <QCommandLineParser>
#include <QDir>

#include "mainwindow.h"
#include "streamserver.h"
#include "renderlib/FileReader.h"
#include "renderlib/renderlib.h"

void preloadFiles(QStringList preloadlist) {
	for (QString s : preloadlist) {
		QFileInfo info(s);
		if (info.exists())
		{
			FileReader::loadOMETiff_4D(info.absoluteFilePath().toStdString(), true);
		}
		else {
			qDebug() << "Could not load " << s;
		}
	}

}

int main(int argc, char *argv[])
{
	//set to true to show windows, or false to run as a console application
	bool gui = false;

	QApplication::setAttribute(Qt::AA_UseDesktopOpenGL, true);
	QApplication a(argc, argv);
	//QDir::setCurrent(QApplication::applicationDirPath() + "/work");

	a.setApplicationName("AICS RENDERSERVER");
	a.setApplicationVersion("0.0.1");

	QCommandLineParser parser;
	parser.setApplicationDescription("Remote rendering service via websockets");
	parser.addHelpOption();
	parser.addVersionOption();
	parser.addPositionalArgument("port", QCoreApplication::translate("main", "Websocket connection port."));
	QCommandLineOption preloadOption(QStringList() << "p" << "preload",
		QCoreApplication::translate("main", "Specify ome-tiff file(s) to pre-load."),
		QCoreApplication::translate("main", "preload"));
	parser.addOption(preloadOption);

	// Process the actual command line arguments given by the user
	parser.process(a);

	const QStringList args = parser.positionalArguments();
	// port is args.at(0)
	const int defaultPort = 1235;
	int port = defaultPort;
	if (args.size() > 0) {
		QString sport = args.at(0);
		bool ok;
		port = sport.toInt(&ok);
		if (!ok) {
			port = defaultPort;
		}
	}

	if (!renderlib::initialize())
	{
		renderlib::cleanup();
		return 0;
	}

	StreamServer *server = new StreamServer(port, false, 0);

	if (gui)
	{
		MainWindow* _ = new MainWindow(server);
		_->resize(512, 512);
		_->show();
	}

	qDebug() << "working directory:" << QDir::current();

	//delete logFile;

	QStringList preloads = parser.values(preloadOption);
	preloadFiles(preloads);

	int result = a.exec();
	renderlib::cleanup();
	return result;
}
