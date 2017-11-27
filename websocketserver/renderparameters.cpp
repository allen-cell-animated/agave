#include "renderparameters.h"

RenderParameters::RenderParameters(QMatrix4x4 modelview, QString type1, QString cell1, QString type2, QString cell2, QList<QVariant> channelvalues, int mode, qreal crossFade, int visibility, qreal mitoFuzziness, const char *format, int quality, bool usingCellServer) :
	modelview(modelview),
	visibility(visibility),
	mitoFuzziness(mitoFuzziness),
	format(format),
	quality(quality),
	type1(type1),
	type2(type2),
	cell1(cell1),
	cell2(cell2),
	channelvalues(channelvalues),
	mode(mode),
	crossFade(crossFade),
	usingCellServer(usingCellServer)
{

}
