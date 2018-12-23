#define NOMINMAX
#include "CudaUtilities.h"
#include <glad/glad.h>

#include "Logging.h"

#include <stdio.h>

CCudaTimer::CCudaTimer(void)
{
  StartTimer();
}

CCudaTimer::~CCudaTimer(void)
{
  cudaEventDestroy(m_EventStart);
  cudaEventDestroy(m_EventStop);
}

void
CCudaTimer::StartTimer(void)
{
  cudaEventCreate(&m_EventStart);
  cudaEventCreate(&m_EventStop);
  cudaEventRecord(m_EventStart, 0);

  m_Started = true;
}

float
CCudaTimer::StopTimer(void)
{
  if (!m_Started)
    return 0.0f;

  cudaEventRecord(m_EventStop, 0);
  cudaEventSynchronize(m_EventStop);

  float TimeDelta = 0.0f;

  cudaEventElapsedTime(&TimeDelta, m_EventStart, m_EventStop);
  cudaEventDestroy(m_EventStart);
  cudaEventDestroy(m_EventStop);

  m_Started = false;

  return TimeDelta;
}

float
CCudaTimer::ElapsedTime(void)
{
  if (!m_Started)
    return 0.0f;

  cudaEventRecord(m_EventStop, 0);
  cudaEventSynchronize(m_EventStop);

  float TimeDelta = 0.0f;

  cudaEventElapsedTime(&TimeDelta, m_EventStart, m_EventStop);

  m_Started = false;

  return TimeDelta;
}

// This function wraps the CUDA Driver API into a template function
template<class T>
inline void
GetCudaAttribute(T* attribute, CUdevice_attribute device_attribute, int device)
{
  CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

  if (CUDA_SUCCESS != error) {
    fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n", error, __FILE__, __LINE__);
    exit(-1);
  }
}

void
HandleCudaError(const cudaError_t CudaError, const char* pDescription /*= ""*/)
{
  if (CudaError == cudaSuccess)
    return;

  LOG_ERROR << "Encountered a critical CUDA error: " << std::string(pDescription) << " "
            << std::string(cudaGetErrorString(CudaError));

  throw std::string("Encountered a critical CUDA error: ") + std::string(pDescription) + std::string(" ") +
    std::string(cudaGetErrorString(CudaError));
}

void
HandleCudaKernelError(const cudaError_t CudaError, const char* pName /*= ""*/)
{
  if (CudaError == cudaSuccess)
    return;

  // try to recover as best we can to minimize chance of catastrophe
  if (CudaError == cudaErrorLaunchFailure) {
    cudaDeviceReset();
  }
  LOG_ERROR << "The '" << std::string(pName)
            << "' kernel caused the following CUDA runtime error: " << std::string(cudaGetErrorString(CudaError));

  throw std::string("The '") + std::string(pName) +
    "' kernel caused the following CUDA runtime error: " + std::string(cudaGetErrorString(CudaError));
}

size_t
GetTotalCudaMemory(void)
{
  size_t Available = 0, Total = 0;
  cudaMemGetInfo(&Available, &Total);
  return Total;
}

size_t
GetAvailableCudaMemory(void)
{
  size_t Available = 0, Total = 0;
  cudaMemGetInfo(&Available, &Total);
  return Available;
}

size_t
GetUsedCudaMemory(void)
{
  size_t Available = 0, Total = 0;
  cudaMemGetInfo(&Available, &Total);
  return Total - Available;
}

int
_ConvertSMVer2Cores(int major, int minor)
{
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = { { 0x10, 8 },   { 0x11, 8 },  { 0x12, 8 },
                                      { 0x13, 8 },   { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
                                      { 0x21, 48 },                // Fermi Generation (SM 2.1) GF10x class
                                      { 0x30, 192 },               // Kepler Generation (SM 3.0) GK10x class
                                      { 0x32, 192 },               // Kepler Generation (SM 3.2) GK10x class
                                      { 0x35, 192 },               // Kepler Generation (SM 3.5) GK11x class
                                      { 0x37, 192 },               // Kepler Generation (SM 3.7) GK21x class
                                      { 0x50, 128 },               // Maxwell Generation (SM 5.0) GM10x class
                                      { 0x52, 128 },               // Maxwell Generation (SM 5.2) GM20x class
                                      { 0x53, 128 },               // Maxwell Generation (SM 5.3) GM20x class
                                      { 0x60, 64 },                // Pascal Generation (SM 6.0) GP100 class
                                      { 0x61, 128 },               // Pascal Generation (SM 6.1) GP10x class
                                      { 0x62, 128 },               // Pascal Generation (SM 6.2) GP10x class
                                      { -1, -1 } };

  int index = 0;
  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }
    index++;
  }
  printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
  return -1;
}

int
GetMaxGigaFlopsDeviceID(void)
{
  int current_device = 0, sm_per_multiproc = 0;
  int max_compute_perf = 0, max_perf_device = 0;
  int device_count = 0, best_SM_arch = 0;
  cudaDeviceProp deviceProp;

  cudaGetDeviceCount(&device_count);
  // Find the best major SM Architecture GPU device
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major > 0 && deviceProp.major < 9999) {
      best_SM_arch = std::max(best_SM_arch, deviceProp.major);
    }
    current_device++;
  }

  // Find the best CUDA capable GPU device
  current_device = 0;
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
      sm_per_multiproc = 1;
    } else {
      sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
    }

    int compute_perf = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
    if (compute_perf > max_compute_perf) {
      // If we find GPU with SM major > 2, search only these
      if (best_SM_arch > 2) {
        // If our device==dest_SM_arch, choose this, or else pass
        if (deviceProp.major == best_SM_arch) {
          max_compute_perf = compute_perf;
          max_perf_device = current_device;
        }
      } else {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    }
    ++current_device;
  }
  return max_perf_device;
}

bool
SetCudaDevice(const int& CudaDeviceID)
{
  const cudaError_t CudaError = cudaSetDevice(CudaDeviceID);

  HandleCudaError(CudaError, "set Cuda device");

  return CudaError == cudaSuccess;
}

void
ResetDevice(void)
{
  HandleCudaError(cudaDeviceReset(), "reset device");
}
