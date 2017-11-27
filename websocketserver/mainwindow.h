#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>

#include "streamserver.h"
#include "renderer.h"

#include <QTextEdit>

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(StreamServer *server);
	~MainWindow();

private:
	QTimer *timer;
	StreamServer *server;
	QTextEdit *output;

	public slots:
	void updateStats();
};

#endif // MAINWINDOW_H
