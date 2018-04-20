#include "CudaConstants.h"

CudaGPU::CudaGPU(int deviceId)
	: deviceId(deviceId)
{
	CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
	gridStrideBlockCount = props.multiProcessorCount * BlockPerSM;
	tier = DetermineGPUTier(props);
}

CudaGPU::GPUTier CudaGPU::DetermineGPUTier(cudaDeviceProp p)
{
	return KEPLER;
}

int CudaGPU::DeviceId() const
{
	return deviceId;
}

std::string CudaGPU::Name() const
{
	return props.name;
}

double CudaGPU::TotalMemoryMB() const
{
	return static_cast<double>(props.totalGlobalMem) / 1000000000.0;
}

double CudaGPU::TotalMemoryGB() const
{
	return static_cast<double>(props.totalGlobalMem) / 1000000.0;
}

size_t CudaGPU::TotalMemory() const
{
	return props.totalGlobalMem;
}

Vector2i CudaGPU::MaxTexture2DSize() const
{
	return Vector2i(props.maxTexture2D[0],
					props.maxTexture2D[1]);
}

int32_t CudaGPU::RecommendedBlockCount() const
{
	return gridStrideBlockCount;
}

std::vector<CudaGPU> CudaSystem::gpus;

bool CudaSystem::Initialize()
{	
	int deviceCount;	
	cudaError err;

	err = cudaGetDeviceCount(&deviceCount);
	if(err == cudaErrorInsufficientDriver)
	{
		return false;
	}
	else if(err == cudaErrorNoDevice)
	{
		return false;
	}

	// All Fine Start Query Devices
	for(int i = 0; i < deviceCount; i++)
	{
		gpus.emplace_back(i);
	}
	return true;
}