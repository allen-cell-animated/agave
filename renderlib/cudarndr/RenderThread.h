#pragma once

#include <QtCore/QMutex>
#include <QtCore/QThread>

#include "Geometry.h"
#include "Variance.h"

class aicsQFrameBuffer
{
public:
	aicsQFrameBuffer(void);
	aicsQFrameBuffer(const aicsQFrameBuffer& Other);
	aicsQFrameBuffer& operator=(const aicsQFrameBuffer& Other);
	virtual ~aicsQFrameBuffer(void);
	void Set(unsigned char* pPixels, const int& Width, const int& Height);
	unsigned char* GetPixels(void) { return m_pPixels; }
	int GetWidth(void) const { return m_Width; }
	int GetHeight(void) const { return m_Height; }
	int GetNoPixels(void) const { return m_NoPixels; }

	QMutex			m_Mutex;

private :
	unsigned char*	m_pPixels;
	int				m_Width;
	int				m_Height;
	int				m_NoPixels;
};

extern aicsQFrameBuffer gFrameBuffer;

class aicsQRenderThread : public QThread
{
	Q_OBJECT

public:
	aicsQRenderThread(const QString& FileName = "", QObject* pParent = NULL);
	aicsQRenderThread(const aicsQRenderThread& Other);
	virtual ~aicsQRenderThread(void);
	aicsQRenderThread& operator=(const aicsQRenderThread& Other);

	void run();

	bool			Load(QString& FileName);

	QString			GetFileName(void) const						{	return m_FileName;		}
	void			SetFileName(const QString& FileName)		{	m_FileName = FileName;	}
	CColorRgbLdr*	GetRenderImage(void) const;
	void			Close(void)									{	m_Abort = true;			}
	void			PauseRendering(const bool& Pause)			{	m_Pause = Pause;		}
	
private:
	QString				m_FileName;
//	CCudaFrameBuffers	m_CudaFrameBuffers;
	CColorRgbLdr*		m_pRenderImage;
	short*				m_pDensityBuffer;
	short*				m_pGradientMagnitudeBuffer;


public:
	bool			m_Abort;
	bool			m_Pause;
	QMutex			m_Mutex;

public:
	QList<int>		m_SaveFrames;
	QString			m_SaveBaseName;

public slots:
	void OnUpdateTransferFunction(void);
	void OnUpdateCamera(void);
	void OnUpdateLighting(void);
	void OnRenderPause(const bool& Pause);
};

// Render thread
extern aicsQRenderThread* gpRenderThread;

void StartRenderThread(QString& FileName);
void KillRenderThread(void);

extern QMutex gSceneMutex;
extern int gCurrentDeviceID;
