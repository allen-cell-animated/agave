#ifndef UTIL_H
#define UTIL_H

#include <QOpenGLShader>
#include <qopenglshaderprogram.h>
#include <QByteArray>
#include <QVariant>
#include <cmath>

#define PI 3.14159265358979323846264338327950

#if QT_VERSION >= 0x050000
#define MOUSE_POS localPos
#else
#define MOUSE_POS posF
#endif

//#include "marion_global.h"

template <class T> class VPointer
{
public:
	static T* toPointer(QVariant v)
	{
		return  (T *)v.value<void *>();
	}

	static QVariant toVariant(T* pointer)
	{
		return qVariantFromValue((void *)pointer);
	}
};

class Util
{
public:
	Util();
	static QOpenGLShaderProgram *compileShader(QByteArray computeShaderSource);
	static QOpenGLShaderProgram *compileShader(QByteArray vertexShaderSource, QByteArray fragmentShaderSource);
	static QOpenGLShaderProgram *compileShader(QByteArray vertexShaderSource, QByteArray geometryShaderSource, QByteArray fragmentShaderSource);
	static QOpenGLShaderProgram *compileShader(QByteArray vertexShaderSource, QByteArray tcsShaderSource, QByteArray tesShaderSource, QByteArray geometryShaderSource, QByteArray fragmentShaderSource);

	static QByteArray loadSource(QString fileNameOrSource, bool loadFromDisk);
	static QOpenGLShaderProgram *loadAndCompileShader(QString computeFileName);
	static QOpenGLShaderProgram *loadAndCompileShader(QString vsFileName, QString fsFileName, bool loadVS = true, bool loadFS = true);
	static QOpenGLShaderProgram *loadAndCompileShader(QString vsFileName, QString gsFileName, QString fsFileName, bool loadVS = true, bool loadGS = true, bool loadFS = true);
	static QOpenGLShaderProgram *loadAndCompileShader(QString vsFileName, QString tcsFileName, QString tesFileName, QString gsFileName, QString fsFileName,
		bool loadVS = true, bool loadTCS = true, bool loadTES = true, bool loadGS = true, bool loadFS = true);

	//math
	static qreal rrandom();
	static qreal mix(qreal a0, qreal a1, qreal x);
	static qreal mix(qreal a0, qreal a1, qreal a2, qreal a3, qreal x);
	static qreal mixCos(qreal a0, qreal a1, qreal x);
	static QPointF mixPointF(QPointF p0, QPointF p1, qreal x);
	static QPointF mixPointF(QPointF p0, QPointF p1, QPointF p2, QPointF p3, qreal x);
	static QPointF mixPointFCos(QPointF p0, QPointF p1, qreal x);
	static QVector2D mixVector2D(QVector2D p0, QVector2D p1, qreal x);
	static QVector2D mixVector2D(QVector2D p0, QVector2D p1, QVector2D p2, QVector2D p3, qreal x);
	static QVector3D mixVector3D(QVector3D p0, QVector3D p1, qreal x);
	static QVector3D mixVector3D(QVector3D p0, QVector3D p1, QVector3D p2, QVector3D p3, qreal x);

};

#endif // UTIL_H
