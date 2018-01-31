#pragma once

#include "Buffer.cuh"

#include "Geometry.h"
#include "Scene.h"

class CCudaView
{
public:
	CCudaView(void) :
		m_Resolution(0, 0),
		m_RunningEstimateXyza(),
		m_FrameEstimateXyza(),
		m_FrameBlurXyza(),
		m_EstimateRgbaLdr(),
		m_DisplayEstimateRgbLdr(),
		m_RandomSeeds1(),
		m_RandomSeeds2()
	{
	}

	~CCudaView(void)
	{
		Free();
	}

	CCudaView(const CCudaView& Other)
	{
		*this = Other;
	}

	CCudaView& operator=(const CCudaView& Other)
	{
		m_Resolution			= Other.m_Resolution;
		m_RunningEstimateXyza	= Other.m_RunningEstimateXyza;
		m_FrameEstimateXyza		= Other.m_FrameEstimateXyza;
		m_FrameBlurXyza			= Other.m_FrameBlurXyza;
		m_EstimateRgbaLdr		= Other.m_EstimateRgbaLdr;
		m_DisplayEstimateRgbLdr	= Other.m_DisplayEstimateRgbLdr;
		m_RandomSeeds1			= Other.m_RandomSeeds1;
		m_RandomSeeds2			= Other.m_RandomSeeds2;

		return *this;
	}

	void Resize(const CResolution2D& Resolution)
	{
		if (m_Resolution == Resolution)
			return;

		m_Resolution = Resolution;

		m_RunningEstimateXyza.Resize(m_Resolution);
		m_FrameEstimateXyza.Resize(m_Resolution);
		m_FrameBlurXyza.Resize(m_Resolution);
		m_EstimateRgbaLdr.Resize(m_Resolution);
		m_DisplayEstimateRgbLdr.Resize(m_Resolution);
		m_RandomSeeds1.Resize(m_Resolution);
		m_RandomSeeds2.Resize(m_Resolution);
	}

	void Reset(void)
	{
//		m_RunningEstimateXyza.Reset();
		m_FrameEstimateXyza.Reset();
//		m_FrameBlurXyza.Reset();
		m_EstimateRgbaLdr.Reset();
		m_DisplayEstimateRgbLdr.Reset();
//		m_RandomSeeds1.Reset();
//		m_RandomSeeds2.Reset();
	}

	void Free(void)
	{
		m_RunningEstimateXyza.Free();
		m_FrameEstimateXyza.Free();
		m_FrameBlurXyza.Free();
		m_EstimateRgbaLdr.Free();
		m_DisplayEstimateRgbLdr.Free();
		m_RandomSeeds1.Free();
		m_RandomSeeds2.Free();

		m_Resolution.Set(Vec2i(0, 0));
	}

	HO int GetWidth(void) const
	{
		return m_Resolution.GetResX();
	}

	HO int GetHeight(void) const
	{
		return m_Resolution.GetResY();
	}

	CResolution2D						m_Resolution;
	CCudaBuffer2D<CColorXyza, true>		m_RunningEstimateXyza;
	CCudaBuffer2D<CColorXyza, true>		m_FrameEstimateXyza;
	CCudaBuffer2D<CColorXyza, true>		m_FrameBlurXyza;
	CCudaBuffer2D<CColorRgbaLdr, true>	m_EstimateRgbaLdr;
	CCudaBuffer2D<CColorRgbLdr, false>	m_DisplayEstimateRgbLdr;
	CCudaRandomBuffer2D					m_RandomSeeds1;
	CCudaRandomBuffer2D					m_RandomSeeds2;
};