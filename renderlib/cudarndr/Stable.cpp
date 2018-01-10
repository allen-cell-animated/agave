#include "Stable.h"

QString GetOpenFileName(const QString& Caption, const QString& Filter, const QString& Icon)
{
	QFileDialog FileDialog;

	FileDialog.setWindowTitle(Caption);
	FileDialog.setNameFilter(Filter);
	FileDialog.setOption(QFileDialog::DontUseNativeDialog, true);
	FileDialog.setWindowIcon(Icon.isEmpty() ? GetIcon("disk") : GetIcon(Icon));

	if (FileDialog.exec() == QMessageBox::Rejected)
		return "";

	return FileDialog.selectedFiles().value(0);
}

QString GetSaveFileName(const QString& Caption, const QString& Filter, const QString& Icon)
{
	QFileDialog FileDialog;

	FileDialog.setWindowTitle(Caption);
	FileDialog.setNameFilter(Filter);
	FileDialog.setOption(QFileDialog::DontUseNativeDialog, true);
	FileDialog.setWindowIcon(Icon.isEmpty() ? GetIcon("disk") : GetIcon(Icon));
	FileDialog.setAcceptMode(QFileDialog::AcceptSave);

	if (FileDialog.exec() == QMessageBox::Rejected)
		return "";

	return FileDialog.selectedFiles().value(0);
}

void SaveImage(const unsigned char* pImageBuffer, const int& Width, const int& Height, QString FilePath /*= ""*/)
{
	if (!pImageBuffer)
	{
		BOOST_LOG_TRIVIAL(error) << "Can't save image, buffer is empty";
		return;
	}

	if (FilePath.isEmpty())
		FilePath = GetSaveFileName("Save Image", "PNG Files (*.png)", "image-export");

	if (!FilePath.isEmpty())
	{
		QImage* pTempImage = new QImage(pImageBuffer, Width, Height,  QImage::Format_RGB888);

		if (!pTempImage->save(FilePath, "PNG") )
			BOOST_LOG_TRIVIAL(info) << "Unable to save image";
		else
			BOOST_LOG_TRIVIAL(info) << FilePath.toStdString() << " saved";

		delete pTempImage;
	}
	else
	{
		BOOST_LOG_TRIVIAL(error) << "Can't save image, file path is empty";
	}
}
