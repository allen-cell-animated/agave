#include "util.h"
#include <QFile>
#include <QXmlStreamReader>
#include <QMessageBox>

Util::Util()
{
}

QOpenGLShaderProgram *Util::compileShader(QByteArray computeShaderSource)
{
	QOpenGLShader *shader = new QOpenGLShader(QOpenGLShader::Compute);
	shader->compileSourceCode(computeShaderSource);

	QOpenGLShaderProgram *program = new QOpenGLShaderProgram;
	program->addShader(shader);
	program->link();
	program->bind();

	return program;
}

QOpenGLShaderProgram *Util::compileShader(QByteArray vertexShaderSource, QByteArray fragmentShaderSource)
{
	return Util::compileShader(vertexShaderSource, QByteArray(), fragmentShaderSource);
}

QOpenGLShaderProgram *Util::compileShader(QByteArray vertexShaderSource, QByteArray geometryShaderSource, QByteArray fragmentShaderSource)
{
	return Util::compileShader(vertexShaderSource, QByteArray(), QByteArray(), geometryShaderSource, fragmentShaderSource);
}

QOpenGLShaderProgram *Util::compileShader(QByteArray vertexShaderSource , QByteArray tcsShaderSource, QByteArray tesShaderSource, QByteArray geometryShaderSource, QByteArray fragmentShaderSource)
{
	bool geometry = geometryShaderSource.length() > 0;
	bool tcs = tcsShaderSource.length() > 0;
	bool tes = tesShaderSource.length() > 0;
	QOpenGLShader *vShader = new QOpenGLShader(QOpenGLShader::Vertex);
	QOpenGLShader *tcShader = tcs ? new QOpenGLShader(QOpenGLShader::TessellationControl) : 0;
	QOpenGLShader *teShader = tes ? new QOpenGLShader(QOpenGLShader::TessellationEvaluation) : 0;
	QOpenGLShader *gShader = geometry ? new QOpenGLShader(QOpenGLShader::Geometry) : 0;
	QOpenGLShader *fShader = new QOpenGLShader(QOpenGLShader::Fragment);

	/*QGLShader *vShader = new QGLShader(QGLShader::Vertex);
	QGLShader *gShader = geometry ? new QGLShader(QGLShader::Geometry) : 0;
	QGLShader *fShader = new QGLShader(QGLShader::Fragment);*/

	vShader->compileSourceCode(vertexShaderSource);
	if (tcs) 
		tcShader->compileSourceCode(tcsShaderSource);
	if (tes) 
		teShader->compileSourceCode(tesShaderSource);
	if (geometry) 
		gShader->compileSourceCode(geometryShaderSource);
	fShader->compileSourceCode(fragmentShaderSource);

	QOpenGLShaderProgram *shader;
	shader = new QOpenGLShaderProgram();
	shader->addShader(vShader);
	if (tcs)
		shader->addShader(tcShader);
	if (tes)
		shader->addShader(teShader);
	if (geometry) 
		shader->addShader(gShader);
	shader->addShader(fShader);

	bool ok = shader->link();

	if (ok)
	{

	}
	else
	{
		//QMessageBox::critical(0, "Shader Error", "Shader not linked. " + QString(fragmentShaderSource));
		qDebug() << "Shader not linked:";
		qDebug() << QString(fragmentShaderSource);
	}
	shader->bind();

	delete vShader;
	if (geometry)
		delete gShader;
	if (tcs)
		delete tcShader;
	if (tes)
		delete teShader;
	delete fShader;

	if (!shader->hasOpenGLShaderPrograms())
	{
		return 0;
	}

	return shader;
}

QByteArray Util::loadSource(QString fileNameOrSource, bool loadFromDisk)
{
	QByteArray sourceArray;

	if (loadFromDisk)
	{
		QFile *sourceFile = new QFile(fileNameOrSource);
		sourceFile->open(QIODevice::ReadOnly);
		sourceArray = sourceFile->readAll();
		sourceFile->close();
		delete sourceFile;
	}
	else
	{
		sourceArray.append(fileNameOrSource);
	}

	return sourceArray;
}

QOpenGLShaderProgram *Util::loadAndCompileShader(QString computeFileName)
{
	QByteArray computeSourceArray;
	computeSourceArray = Util::loadSource(computeFileName, true);

	qDebug() << "compiling" << computeFileName << "...";

	QOpenGLShaderProgram *shader = 0;

	shader = Util::compileShader(computeSourceArray);

	return shader;
}

QOpenGLShaderProgram *Util::loadAndCompileShader(QString vsFileName, QString gsFileName, QString fsFileName,
	bool loadVS, bool loadGS, bool loadFS)
{
	return Util::loadAndCompileShader(vsFileName, "", "", gsFileName, fsFileName, loadVS, false, false, loadGS, loadFS);
}

QOpenGLShaderProgram *Util::loadAndCompileShader(QString vsFileName, QString fsFileName, bool loadVS, bool loadFS)
{
	return Util::loadAndCompileShader(vsFileName, "", fsFileName, loadVS, false, loadFS);
}

//QOpenGLShaderProgram *Util::loadAndCompileShader(QString vsFileName, QString gsFileName, QString fsFileName, bool loadVS, bool loadGS, bool loadFS)
QOpenGLShaderProgram *Util::loadAndCompileShader(QString vsFileName, QString tcsFileName, QString tesFileName, QString gsFileName, QString fsFileName, 
	bool loadVS, bool loadTCS, bool loadTES, bool loadGS, bool loadFS)
{
	bool vertex = vsFileName.length() > 0;
	bool tcs = vsFileName.length() > 0;
	bool tes = vsFileName.length() > 0;
	bool geometry = gsFileName.length() > 0;
	bool fragment = fsFileName.length() > 0;

	QByteArray vsSourceArray;
	QByteArray tcsSourceArray;
	QByteArray tesSourceArray;
	QByteArray gsSourceArray;
	QByteArray fsSourceArray;

	if (vertex)
	{
		vsSourceArray = Util::loadSource(vsFileName, loadVS);
	}
	if (tcs)
	{
		tcsSourceArray = Util::loadSource(tcsFileName, loadTCS);
	}
	if (tes) 
	{
		tesSourceArray = Util::loadSource(tesFileName, loadTES);
	}
	if (geometry)
	{
		gsSourceArray = Util::loadSource(gsFileName, loadGS);
	}
	if (fragment)
	{
		fsSourceArray = Util::loadSource(fsFileName, loadFS);
	}


	/*

	if (loadVS)
	{
		QFile *vsSource = new QFile(vsFileName);
		vsSource->open(QIODevice::ReadOnly);
		vsSourceArray = vsSource->readAll();
		vsSource->close();
		delete vsSource;
	}
	else
	{
		vsSourceArray.append(vsFileName);
	}

	if (geometry)
	{
		if (loadGS)
		{
			QFile *gsSource = new QFile(gsFileName);
			gsSource->open(QIODevice::ReadOnly);
			gsSourceArray = gsSource->readAll();
			gsSource->close();
			delete gsSource;
		}
		else
		{
			gsSourceArray.append(gsFileName);
		}
	}

	if (loadFS)
	{
		QFile *fsSource = new QFile(fsFileName);
		fsSource->open(QIODevice::ReadOnly);
		fsSourceArray = fsSource->readAll();
		fsSource->close();
		delete fsSource;
	}
	else
	{
		fsSourceArray.append(fsFileName);
	}*/

    qDebug() << "compiling" << vsFileName << fsFileName << "...";

	QOpenGLShaderProgram *shader = 0;
	if (tcs && tes && geometry)
		shader = Util::compileShader(vsSourceArray, tcsSourceArray, tesSourceArray, gsSourceArray, fsSourceArray);
	else if (geometry && !tcs && !tes)
		shader = Util::compileShader(vsSourceArray, gsSourceArray, fsSourceArray);
	else
		shader = Util::compileShader(vsSourceArray, fsSourceArray);

	return shader;
}



//math
qreal Util::rrandom()
{
	return (qreal) (rand() % RAND_MAX) / (qreal) RAND_MAX;
}

qreal Util::mix(qreal a0, qreal a1, qreal x)
{
	return a0 * (1.0 - x) + a1 * x;
}

qreal Util::mixCos(qreal a0, qreal a1, qreal x)
{
	qreal f = (1.0 - cos(x * PI)) * 0.5;
	return a0 * (1.0 - f) + a1 * f;
}

qreal Util::mix(qreal a0, qreal a1, qreal a2, qreal a3, qreal x)
{
	qreal p,q,r,s;
	p = (a3 - a2) - (a0 - a1);
	q = (a0 - a1) - p;
	r = a2 - a0;
	s = a1;

	return p*x*x*x + q*x*x + r*x + s;
}

QPointF Util::mixPointF(QPointF p0, QPointF p1, qreal x)
{
	return QPointF(
				mix(p0.x(), p1.x(), x),
				mix(p0.y(), p1.y(), x));
}

QPointF Util::mixPointF(QPointF p0, QPointF p1, QPointF p2, QPointF p3, qreal x)
{
	return QPointF(
				mix(p0.x(), p1.x(), p2.x(), p3.x(), x),
				mix(p0.y(), p1.y(), p2.y(), p3.y(), x));
}

QPointF Util::mixPointFCos(QPointF p0, QPointF p1, qreal x)
{
	return QPointF(
				mix(p0.x(), p1.x(), x),
				mixCos(p0.y(), p1.y(), x));
}

QVector2D Util::mixVector2D(QVector2D p0, QVector2D p1, qreal x)
{
	return QVector2D(
				mix(p0.x(), p1.x(), x),
				mix(p0.y(), p1.y(), x));
}

QVector2D Util::mixVector2D(QVector2D p0, QVector2D p1, QVector2D p2, QVector2D p3, qreal x)
{
	return QVector2D(
				mix(p0.x(), p1.x(), p2.x(), p3.x(), x),
				mix(p0.y(), p1.y(), p2.y(), p3.y(), x));
}

QVector3D Util::mixVector3D(QVector3D p0, QVector3D p1, qreal x)
{
	return QVector3D(
				mix(p0.x(), p1.x(), x),
				mix(p0.y(), p1.y(), x),
				mix(p0.z(), p1.z(), x));
}

QVector3D Util::mixVector3D(QVector3D p0, QVector3D p1, QVector3D p2, QVector3D p3, qreal x)
{
	return QVector3D(
				mix(p0.x(), p1.x(), p2.x(), p3.x(), x),
				mix(p0.y(), p1.y(), p2.y(), p3.y(), x),
				mix(p0.z(), p1.z(), p2.z(), p3.z(), x));
}
