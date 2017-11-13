#include "renderparameters.h"

RenderParameters::RenderParameters(QMatrix4x4 modelview, int visibility, qreal mitoFuzziness, const char *format, int quality) :
	modelview(modelview),
	visibility(visibility),
	mitoFuzziness(mitoFuzziness),
	format(format),
	quality(quality)
{

}
