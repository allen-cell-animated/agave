#include "mainwindow.h"

MainWindow::MainWindow()
{
	this->setWindowTitle(qAppName() + " " + qApp->applicationVersion());

	/*canvas = new Canvas();
	this->setCentralWidget(canvas);*/

	//canvas->setFocusPolicy(Qt::StrongFocus);
	//oglWindow->grabKeyboard();*/

	//connect(canvas, SIGNAL(kill()), this, SLOT(close()));

	this->output = new QTextEdit(this);

	this->setCentralWidget(this->output);

	this->server = new StreamServer(1234, false, this);
	connect(this->server, &StreamServer::closed, qApp, &QApplication::quit);

	this->timer = new QTimer(this);
	connect(this->timer, SIGNAL(timeout()), this, SLOT(updateStats()));
	this->timer->start(1000);
}

MainWindow::~MainWindow()
{

}

void MainWindow::updateStats()
{
	this->output->clear();

	QList<QWebSocket *> clients = this->server->getClients();
	QList<int> loads = this->server->getThreadsLoad();
	QList<int> requests = this->server->getThreadsRequestCount();

	QString text = "Active connections: " + QString::number(this->server->getClientsCount()) + "\n";

	for (int i = 0; i < this->server->getClientsCount(); i++)
	{
		text += "   client #" + QString::number(i) + ": " +
				clients[i]->peerName() + "(" +
				clients[i]->peerAddress().toString() + ":" + QString::number(clients[i]->peerPort()) + ")\n";
	}


	text += "\nRendering threads: " + QString::number(this->server->getThreadsCount()) + "\n";

	for (int i = 0; i < this->server->getThreadsCount(); i++)
	{
		text += "   thread #" + QString::number(i) + ": " + QString::number(requests[i]) + " requests, " + QString::number(loads[i]) + "ms\n";
	}

	text += "\n\n" + this->server->getTimingsString();

	this->output->setPlainText(text);
}
