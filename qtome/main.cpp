#include "qtome.h"

#include "renderlib/Logging.h"
#include "renderlib/renderlib.h"
#include "renderlib/version.h"

#include <QtWidgets/QApplication>

int
main(int argc, char* argv[])
{
  QApplication a(argc, argv);
  a.setAttribute(Qt::AA_EnableHighDpiScaling);
  a.setOrganizationName("AICS");
  a.setOrganizationDomain("allencell.org");
  a.setApplicationName("GPU Volume Explorer");
  a.setApplicationVersion(AICS_VERSION_STRING);
  LOG_INFO << a.organizationName().toStdString() << " " << a.applicationName().toStdString() << " " << a.applicationVersion().toStdString();

  if (!renderlib::initialize()) {
    renderlib::cleanup();
    return 0;
  }

  qtome w;
  w.show();
  int result = a.exec();

  renderlib::cleanup();

  return result;
}
