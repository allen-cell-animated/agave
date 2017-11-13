#pragma once

#include <string>
#include <vector>

class aicsCudaDevice
{
public:
	aicsCudaDevice(void){};

	aicsCudaDevice& aicsCudaDevice::operator=(const aicsCudaDevice& Other)
	{
		m_ID				= Other.m_ID;
		m_Name				= Other.m_Name;
		m_Capability		= Other.m_Capability;
		m_GlobalMemory		= Other.m_GlobalMemory;
		m_MemoryClockRate	= Other.m_MemoryClockRate;
		m_NoMultiProcessors	= Other.m_NoMultiProcessors;
		m_GpuClockSpeed		= Other.m_GpuClockSpeed;
		m_RegistersPerBlock	= Other.m_RegistersPerBlock;

		return *this;
	}

	int			m_ID;
	std::string		m_Name;
	std::string		m_Capability;
	std::string		m_GlobalMemory;
	std::string		m_MemoryClockRate;
	std::string		m_NoMultiProcessors;
	std::string		m_GpuClockSpeed;
	std::string		m_RegistersPerBlock;
};


class DeviceSelector
{
public:
	DeviceSelector();

	void EnumerateDevices(void);
	void AddDevice(const aicsCudaDevice& Device);
	bool GetDevice(const int& DeviceID, aicsCudaDevice& CudaDevice);
	void OnOptimalDevice(void);
	void OnCudaDeviceChanged(void);

	std::vector<aicsCudaDevice> listOfPairs;

	int _selectedDevice;
	int _selectedDeviceID;
};
