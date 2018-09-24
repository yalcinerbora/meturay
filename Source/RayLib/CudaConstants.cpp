#include "CudaConstants.h"
#include "TracerError.h"

CudaGPU::CudaGPU(int deviceId)
	: deviceId(deviceId)
{
	CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
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

uint32_t CudaGPU::SMCount() const
{
	return static_cast<uint32_t>(props.multiProcessorCount);
}

uint32_t CudaGPU::RecommendedBlockCountPerSM(void* kernelFunc,
											 uint32_t threadsPerBlock,
											 uint32_t sharedSize) const
{
	int32_t numBlocks = 0;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
															 kernelFunc,
															 threadsPerBlock,
															 sharedSize));

	return static_cast<uint32_t>(numBlocks);
}



std::vector<CudaGPU> CudaSystem::gpus;

TracerError CudaSystem::Initialize()
{	
	int deviceCount;	
	cudaError err;

	err = cudaGetDeviceCount(&deviceCount);
	if(err == cudaErrorInsufficientDriver)
	{
		return TracerError::CUDA_OLD_DRIVER;
	}
	else if(err == cudaErrorNoDevice)
	{
		return TracerError::CUDA_NO_DEVICE;
	}

	// All Fine Start Query Devices
	for(int i = 0; i < deviceCount; i++)
	{
		gpus.emplace_back(i);
	}
	return TracerError::OK;
}