#include "CudaConstants.h"
#include "RayLib/TracerError.h"

CudaGPU::CudaGPU(int deviceId)
	: deviceId(deviceId)
{
	CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
	tier = DetermineGPUTier(props);
}

//CudaGPU::CudaGPU(CudaGPU&& other)
//	other
//{
//
//}

CudaGPU::GPUTier CudaGPU::DetermineGPUTier(cudaDeviceProp p)
{
	if(p.major == 3) return GPU_KEPLER;
	else if(p.major == 5) return GPU_MAXWELL;	
	else if(p.major == 6) return GPU_PASCAL;
	else if(p.major >= 7) return GPU_TURING_VOLTA;
	else return GPU_UNSUPPORTED;
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

cudaStream_t CudaGPU::DetermineStream(uint32_t requiredSMCount) const
{
	if(requiredSMCount >= (SMCount() / 2))
		return largeWorkList.UseGroup();
	else if(requiredSMCount >= (SMCount() / 4))
		return mediumWorkList.UseGroup();
	else// if(requiredSMCount >= (SMCount() / 8))
		return smallWorkList.UseGroup();
}

METU_SHARED_TRACER_ENTRY_POINT std::vector<CudaGPU> CudaSystem::systemGPUs;

TracerError CudaSystem::Initialize()
{
	return Initialize(systemGPUs);
}

TracerError CudaSystem::Initialize(std::vector<CudaGPU>& gpus)
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

	// Strip unsupported processors

	return TracerError::OK;
}

uint32_t CudaSystem::DetermineGridStrideBlock(int gpuIndex,
											  uint32_t sharedMemSize,
											  uint32_t threadCount,
											  size_t workCount,
											  void* func)
{
	const CudaGPU& gpu = systemGPUs[gpuIndex];
	
	uint32_t blockPerSM = gpu.RecommendedBlockCountPerSM(func, threadCount,
														 sharedMemSize);
	// Only call enough SM
	uint32_t totalRequiredBlocks = static_cast<uint32_t>((workCount + (threadCount - 1)) / threadCount);
	uint32_t requiredSMCount = totalRequiredBlocks / blockPerSM;
	uint32_t smCount = std::min(gpu.SMCount(), requiredSMCount);
	uint32_t blockCount = smCount * blockPerSM;
	return blockCount;
}

const std::vector<size_t> CudaSystem::GridStrideMultiGPUSplit(size_t workCount,
															  uint32_t threadCount,
															  uint32_t sharedMemSize,
															  void* f)
{

	std::vector<size_t> workPerGPU;
	// Split work into all GPUs
	uint32_t totalAvailBlocks = 0;
	for(CudaGPU g : systemGPUs)
	{
		uint32_t blockPerSM = g.RecommendedBlockCountPerSM(f, threadCount, sharedMemSize);
		totalAvailBlocks += blockPerSM * g.SMCount();
	}

	size_t workDispatched = 0;
	//const uint32_t totalWorkPerSM = (workCount + totalSMs - 1) / totalSMs;
	for(CudaGPU g : systemGPUs)
	{
		uint32_t blockPerSM = g.RecommendedBlockCountPerSM(f, threadCount, sharedMemSize);
		uint32_t gpuBlocks = blockPerSM * g.SMCount();
		size_t gpuWorkCount = StaticThreadPerBlock1D * gpuBlocks;
		gpuWorkCount = std::min(gpuWorkCount, workCount - workDispatched);
		workDispatched += gpuWorkCount;
		workPerGPU.emplace_back(gpuWorkCount);
	}
	return std::move(workPerGPU);
}