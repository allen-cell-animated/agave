#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "renderer.h"
#include "streamserver.h"

#include <QMainWindow>
#include <QTextEdit>
#include <QTimer>

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  MainWindow(StreamServer* server);
  ~MainWindow();

private:
  QTimer* timer;
  StreamServer* server;
  QTextEdit* output;

public slots:
  void updateStats();
};

#endif // MAINWINDOW_H
