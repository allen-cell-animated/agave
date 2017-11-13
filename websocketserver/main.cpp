#include <QApplication>
#include <QDir>

#include "mainwindow.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QDir::setCurrent(QApplication::applicationDirPath() + "/work");

	a.setApplicationName("CellServer");
	a.setApplicationVersion("0.7.3 1004");

	MainWindow _;
	_.resize(512, 512);
	_.show();

	qDebug() << "working directory:" << QDir::current();

	return a.exec();
}
