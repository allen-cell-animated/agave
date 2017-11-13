#ifndef RENDERPARAMETERS_H
#define RENDERPARAMETERS_H

#include <QtGlobal>
#include <QMatrix4x4>

class RenderParameters
{
public:
	RenderParameters(QMatrix4x4 modelview, int visibility, qreal mitoFuzziness, const char *format, int quality = -1);

	QMatrix4x4 modelview;
	int visibility;
	qreal mitoFuzziness;
	const char *format;
	int quality;
};

#endif // RENDERPARAMETERS_H
