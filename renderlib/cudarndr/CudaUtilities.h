#pragma once

#include "glad/glad.h"

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <surface_types.h>

class CCudaTimer
{
public:
	CCudaTimer(void);
	virtual ~CCudaTimer(void);

	void	StartTimer(void);
	float	StopTimer(void);
	float	ElapsedTime(void);

private:
	bool			m_Started;
	cudaEvent_t 	m_EventStart;
	cudaEvent_t 	m_EventStop;
};

void HandleCudaError(const cudaError_t CudaError, const char* pDescription = "");
void HandleCudaKernelError(const cudaError_t CudaError, const char* pName = "");
size_t GetTotalCudaMemory(void);
size_t GetAvailableCudaMemory(void);
size_t GetUsedCudaMemory(void);
int GetMaxGigaFlopsDeviceID(void);
bool SetCudaDevice(const int& CudaDeviceID);
void ResetDevice(void);
