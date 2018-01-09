#include "qtome.h"
#include "renderlib/renderlib.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	
	if (!renderlib::initialize())
	{
		renderlib::cleanup();
		return 0;
	}

	qtome w;
	w.show();
	int result = a.exec();

	renderlib::cleanup();

	return result;
}
