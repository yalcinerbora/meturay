#pragma once
/**
*/

#include <vector>
#include "DeviceMemory.h"
#include "RNGStructs.h"

class RNGMemory
{
	private:
		DeviceMemory						memRandom;
		std::vector<RNGGMem>				randomStacks;

	protected:
	public:
		// Constructors & Destructor
											RNGMemory() = default;
											RNGMemory(uint32_t seed);
											RNGMemory(const RNGMemory&) = delete;
											RNGMemory(RNGMemory&&) = default;
		RNGMemory&							operator=(const RNGMemory&) = delete;
		RNGMemory&							operator=(RNGMemory&&) = default;
											~RNGMemory() = default;

		RNGGMem								RNGData(uint32_t gpuId);
		ConstRNGGMem						RNGData(uint32_t gpuId) const;
		uint32_t							SharedMemorySize(uint32_t threadPerBlock);
};