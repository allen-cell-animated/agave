#include "cgiparser.h"

CgiParser::CgiParser(int argc, char* argv[])
{
  // environment
  QProcessEnvironment e(QProcessEnvironment::systemEnvironment());
  foreach (QString key, e.keys()) {
    this->m_env.insert(key, e.value(key));
  }

  // load the script
  this->m_script = "";

  if (argc > 1) {
    QFile scriptFile(argv[1]);
    scriptFile.open(QIODevice::ReadOnly);
    this->m_script = scriptFile.readAll();
    scriptFile.close();
  }

  // get request
  this->m_getString = QString(qgetenv("QUERY_STRING"));
  this->m_get = this->urlDecode(this->m_getString);

  // cookie request
  this->m_cookieString = QString(qgetenv("HTTP_COOKIE"));
  this->m_cookie = this->urlDecode(this->m_cookieString);

  // post request
  this->m_postString = "";
  QTextStream in(stdin);

  QString line;
  do {
    line = in.readLine();

    this->m_postString += line + "\n";
  } while (!line.isNull());

  this->m_post = this->urlDecode(this->m_postString);
}

QHash<QString, QString>
CgiParser::urlDecode(QString urlEncoded)
{
  QHash<QString, QString> hash;
  QStringList parts = urlEncoded.split("&");
  foreach (QString part, parts) {
    QStringList nameValue = part.split("=");

    if (nameValue.size() > 1) {
      hash.insert(nameValue[0].trimmed(), nameValue[1].trimmed());
    } else if (nameValue.size() > 0) {
      hash.insert(nameValue[0].trimmed(), QString());
    }
  }

  return hash;
}
