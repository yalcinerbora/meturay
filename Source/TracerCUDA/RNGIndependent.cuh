#pragma once
/**
*/

#include <map>
#include "DeviceMemory.h"
#include "RNGenerator.h"
#include <curand_kernel.h>

class CudaGPU;
class CudaSystem;

class RNGIndependentGPU : public RNGeneratorGPUI
{
    private:
        curandStateMRG32k3a_t       rState;

    protected:
    public:
        // Constructor
        __device__              RNGIndependentGPU(uint32_t seed,
                                                  uint32_t subsequence);
                                RNGIndependentGPU(const RNGIndependentGPU&) = delete;
        RNGIndependentGPU&      operator=(const RNGIndependentGPU&) = delete;
                                ~RNGIndependentGPU() = default;

        __device__ float        Uniform() override;
        __device__ float        Uniform(float min, float max) override;
        __device__ Vector2f     Uniform2D() override;
        __device__ float        Normal() override;
        __device__ float        Normal(float mean, float stdDev) override;
};

class RNGIndependentCPU : public RNGeneratorCPUI
{
    private:
        DeviceMemory                                    memRandom;
        std::map<const CudaGPU*, RNGeneratorGPUI**>    deviceGenerators;

    protected:
    public:
        // Constructors & Destructor
                            RNGIndependentCPU() = default;
                            RNGIndependentCPU(uint32_t seed,
                                              const CudaSystem&);
                            RNGIndependentCPU(uint32_t seed,
                                              const CudaGPU&);
                            RNGIndependentCPU(const RNGIndependentCPU&) = delete;
                            RNGIndependentCPU(RNGIndependentCPU&&) = default;
        RNGIndependentCPU&  operator=(const RNGIndependentCPU&) = delete;
        RNGIndependentCPU&  operator=(RNGIndependentCPU&&) = default;
                            ~RNGIndependentCPU() = default;

        RNGeneratorGPUI**   GetGPUGenerators(const CudaGPU&) override;
        size_t              UsedGPUMemory() const override;
};



inline size_t RNGIndependentCPU::UsedGPUMemory() const
{
    return memRandom.Size();
}

__device__ inline
RNGIndependentGPU::RNGIndependentGPU(uint32_t seed,
                                     uint32_t subsequence)
{
    curand_init(seed, subsequence, 0, &rState);
}

__device__ inline
float RNGIndependentGPU::Uniform()
{
    // curand returns (0, 1]
    // we need [0, 1) invert it
    float result = 1 - curand_uniform(&rState);
    // TODO: curand rarely returns 1.0f anyways
    // (is it a bug?, or "1.0f - x" is causing the bug)
    // Just if it returns 1.0f return nearest float closest to 1.0f
    result = (result == 1) ? nextafter(result, 0.0f) : result;
    return result;
}

__device__ inline
float RNGIndependentGPU::Uniform(float min, float max)
{
    float result = Uniform() * (min - max) + min;
    return result;
}

__device__ inline
Vector2f RNGIndependentGPU::Uniform2D()
{
    return Vector2f(Uniform(), Uniform());
}

__device__ inline
float RNGIndependentGPU::Normal()
{
    return curand_normal(&rState);
}

__device__ inline
float RNGIndependentGPU::Normal(float mean, float stdDev)
{
    return curand_normal(&rState) * stdDev + mean;
}
