#pragma once
/**
*/

#include <map>
#include "DeviceMemory.h"
#include "RNGStructs.h"

class CudaGPU;
class CudaSystem;

class RNGMemory
{
    private:
        DeviceMemory                        memRandom;
        std::map<const CudaGPU*, RNGGMem>   randomStacks;

    protected:
    public:
        // Constructors & Destructor
                            RNGMemory() = default;
                            RNGMemory(uint32_t seed,
                                      const CudaSystem&);
                            RNGMemory(uint32_t seed,
                                      const CudaGPU&);
                            RNGMemory(const RNGMemory&) = delete;
                            RNGMemory(RNGMemory&&) = default;
        RNGMemory&          operator=(const RNGMemory&) = delete;
        RNGMemory&          operator=(RNGMemory&&) = default;
                            ~RNGMemory() = default;

        RNGGMem             RNGData(const CudaGPU&);

        // Memory Usage
        size_t              UsedGPUMemory() const;
};

inline size_t RNGMemory::UsedGPUMemory() const
{
    return memRandom.Size();
}