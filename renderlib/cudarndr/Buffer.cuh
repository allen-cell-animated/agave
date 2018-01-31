#pragma once

#include "Geometry.h"
#include "CudaUtilities.h"

template<class T, bool Pitched>
class CCudaBuffer2D
{
public:
	CCudaBuffer2D(void) :
		m_Resolution(0, 0),
		m_pData(NULL),
		m_Pitch(0)
	{
	}

	virtual ~CCudaBuffer2D(void)
	{
		Free();
	}

	CCudaBuffer2D(const CCudaBuffer2D& Other)
	{
		*this = Other;
	}

	CCudaBuffer2D& operator=(const CCudaBuffer2D& Other)
	{
		m_Resolution	= Other.m_Resolution;
		m_pData			= Other.m_pData;
		m_Pitch			= Other.m_Pitch;

		return *this;
	}

	void Resize(const CResolution2D& Resolution)
	{
		if (m_Resolution != Resolution)
			Free();

		m_Resolution = Resolution;

		if (GetNoElements() <= 0)
			return;

		if (Pitched)
			HandleCudaError(cudaMallocPitch(&m_pData, &m_Pitch, GetWidth() * sizeof(T), GetHeight()));
		else
			HandleCudaError(cudaMalloc(&m_pData, GetSize()));

		Reset();
	}

	void Reset(void)
	{
		if (GetSize() <= 0)
			return;

		HandleCudaError(cudaMemset(m_pData, 0, GetSize()));
	}

	void Free(void)
	{
		if (m_pData)
		{
			HandleCudaError(cudaFree(m_pData));
			m_pData = NULL;
		}
		
		m_Pitch	= 0;
		m_Resolution.Set(Vec2i(0, 0));
	}

	HO int GetNoElements(void) const
	{
		return m_Resolution.GetNoElements();
	}

	HO int GetSize(void) const
	{
		if (Pitched)
			return m_Resolution.GetResY() * (int)m_Pitch;
		else
			return GetNoElements() * sizeof(T);
	}

	HO T Get(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return T();

		if (Pitched)
			return m_pData[Y * (GetPitch() / sizeof(T)) + X];
		else
			return m_pData[Y * GetWidth() + X];
	}

	HOD T& GetRef(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return T();

		if (Pitched)
			return m_pData[Y * (GetPitch() / sizeof(T)) + X];
		else
			return m_pData[Y * GetWidth() + X];
	}

	HOD T* GetPtr(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return NULL;

		if (Pitched)
			return &m_pData[Y * (GetPitch() / sizeof(T)) + X];
		else
			return &m_pData[Y * GetWidth() + X];
	}

	HO void Set(T& Value, const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return;

		if (Pitched)
			m_pData[Y * (GetPitch() / sizeof(T)) + X] = Value;
		else
			m_pData[Y * GetWidth() + X] = Value;
	}

	HO int GetWidth(void) const
	{
		return m_Resolution.GetResX();
	}

	HO int GetHeight(void) const
	{
		return m_Resolution.GetResY();
	}

	HO int GetPitch(void) const
	{
		if (Pitched)
			return (int)m_Pitch;
		else
			return GetWidth() * sizeof(T);
	}

protected:
	CResolution2D	m_Resolution;
	T*				m_pData;
	size_t			m_Pitch;
};

class CCudaRandomBuffer2D : public CCudaBuffer2D<unsigned int, false>
{
public:
	void Resize(const CResolution2D& Resolution)
	{
		CCudaBuffer2D::Resize(Resolution);

		unsigned int* pSeeds = (unsigned int*)malloc(GetSize());

		memset(pSeeds, 0, GetSize());

		for (int i = 0; i < GetNoElements(); i++)
			pSeeds[i] = rand();

		HandleCudaError(cudaMemcpy(m_pData, pSeeds, GetSize(), cudaMemcpyHostToDevice));

		free(pSeeds);
	}
};