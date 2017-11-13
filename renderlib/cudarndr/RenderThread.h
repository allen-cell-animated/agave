/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
