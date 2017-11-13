#ifndef CGIPARSER_H
#define CGIPARSER_H

#include <QString>
#include <QHash>
#include <QTextStream>
#include <QFile>
#include <QProcessEnvironment>

class CgiParser
{
public:
	CgiParser(int argc, char *argv[]);

	QHash<QString, QString> urlDecode(QString urlEncoded);

	QHash<QString, QString> env;
	QHash<QString, QString> get;
	QHash<QString, QString> post;
	QHash<QString, QString> cookie;

private:
	QString script;
	QString getString;
	QString postString;
	QString cookieString;

};

#endif // CGIPARSER_H

