#include "cgiparser.h"

CgiParser::CgiParser(int argc, char *argv[])
{
	//environment
	QProcessEnvironment e(QProcessEnvironment::systemEnvironment());
	foreach (QString key, e.keys())
	{
		this->env.insert(key, e.value(key));
	}

	//load the script
	this->script = "";

	if (argc > 1)
	{
		QFile scriptFile(argv[1]);
		scriptFile.open(QIODevice::ReadOnly);
		this->script = scriptFile.readAll();
		scriptFile.close();
	}

	//get request
	this->getString = QString(qgetenv("QUERY_STRING"));
	this->get = this->urlDecode(this->getString);

	//cookie request
	this->cookieString = QString(qgetenv("HTTP_COOKIE"));
	this->cookie = this->urlDecode(this->cookieString);

	//post request
	this->postString = "";
	QTextStream in(stdin);

	QString line;
	do {
		line = in.readLine();

		this->postString += line + "\n";
	} while (!line.isNull());

	this->post = this->urlDecode(this->postString);
}

QHash<QString, QString> CgiParser::urlDecode(QString urlEncoded)
{
	QHash<QString, QString> hash;
	QStringList parts = urlEncoded.split("&");
	foreach (QString part, parts)
	{
		QStringList nameValue = part.split("=");

		if (nameValue.size() > 1)
		{
			hash.insert(nameValue[0].trimmed(), nameValue[1].trimmed());
		}
		else if (nameValue.size() > 0)
		{
			hash.insert(nameValue[0].trimmed(), QString());
		}
	}

	return hash;
}
