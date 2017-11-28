#ifndef RENDERPARAMETERS_H
#define RENDERPARAMETERS_H

#include <QtGlobal>
#include <QMatrix4x4>

class RenderParameters
{
public:
	//RenderParameters(QMatrix4x4 modelview, int visibility, qreal mitoFuzziness, const char *format, int quality = -1);
	RenderParameters(QMatrix4x4 modelview, QString type1, QString cell1, QString type2, QString cell2, QList<QVariant> channelvalues, int mode, qreal crossFade, int visibility, qreal mitoFuzziness, const char *format, int quality = -1, bool usingCellServer = true);

	QMatrix4x4 modelview;
	QString type1;
	QString cell1;
	QString type2;
	QString cell2;
	QList<QVariant> channelvalues;
	int mode;
	qreal crossFade;
	bool usingCellServer;


	const char *format;
	int quality;

	int mseDx, mseDy;

	//deprecated?
	int visibility;
	qreal mitoFuzziness;
};

#endif // RENDERPARAMETERS_H
