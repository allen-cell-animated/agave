#ifndef CGIPARSER_H
#define CGIPARSER_H

#include <QFile>
#include <QHash>
#include <QProcessEnvironment>
#include <QString>
#include <QTextStream>

class CgiParser
{
public:
  CgiParser(int argc, char* argv[]);

  QHash<QString, QString> urlDecode(QString urlEncoded);

  QHash<QString, QString> m_env;
  QHash<QString, QString> m_get;
  QHash<QString, QString> m_post;
  QHash<QString, QString> m_cookie;

private:
  QString m_script;
  QString m_getString;
  QString m_postString;
  QString m_cookieString;
};

#endif // CGIPARSER_H
