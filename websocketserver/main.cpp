#include <QApplication>
#include <QDir>

#include "mainwindow.h"
#include "streamserver.h"

int main(int argc, char *argv[])
{
	//set to true to show windows, or false to run as a console application
	bool gui = false;

	QApplication a(argc, argv);
	QDir::setCurrent(QApplication::applicationDirPath() + "/work");

	a.setApplicationName("CellServer");
	a.setApplicationVersion("0.7.5 1001");

	StreamServer *server = new StreamServer(1234, false, 0);

	if (gui)
	{
		MainWindow _(server);
		_.resize(512, 512);
		_.show();
	}

	qDebug() << "working directory:" << QDir::current();

	//delete logFile;

	return a.exec();
}
