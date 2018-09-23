#include "RNGMemory.h"
#include "CudaConstants.h"
#include <random>

RNGMemory::RNGMemory(uint32_t seed)
{
	assert(CudaSystem::GPUList().size() > 0);

	// CPU Mersenne Twister
	std::mt19937 rng;
	rng.seed(seed);

	// Determine GPU
	size_t totalCount = 0;
	for(const auto& gpu : CudaSystem::GPUList())
	{
		totalCount += BlocksPerSM * gpu.SMCount() * StaticThreadPerBlock1D;
	}

	// Actual Allocation
	size_t totalSize = totalCount * sizeof(uint32_t);
	memRandom = std::move(DeviceMemory(totalSize));
	uint32_t* d_ptr = static_cast<uint32_t*>(memRandom);

	size_t totalOffset = 0;
	for(const auto& gpu : CudaSystem::GPUList())
	{
		randomStacks.emplace_back(RNGGMem{d_ptr + totalOffset});
		totalOffset += BlocksPerSM * gpu.SMCount() * StaticThreadPerBlock1D;
	}
	assert(totalCount == totalOffset);

	// Init all seeds
	std::vector<uint32_t> seeds(totalCount);
	for(size_t i = 0; i < totalCount; i++)
	{
		d_ptr[i] = rng();
	}
}

RNGGMem RNGMemory::RNGData(uint32_t gpuId)
{
	return randomStacks[gpuId];
}

ConstRNGGMem RNGMemory::RNGData(uint32_t gpuId) const
{
	return randomStacks[gpuId];
}

size_t RNGMemory::SharedMemorySize(uint32_t gpuId)
{
	return StaticThreadPerBlock1D * sizeof(uint32_t);
}