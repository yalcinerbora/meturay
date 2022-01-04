#pragma once

#include <map>
#include "DeviceMemory.h"
#include "RNGenerator.h"
#include <curand_kernel.h>

class CudaGPU;
class CudaSystem;

class RNGSobolGPU final : public RNGeneratorGPUI
{
    private:
        curandStateScrambledSobol32_t rState;

    public:
        // Constructor
        __device__              RNGSobolGPU(curandDirectionVectors32_t,
                                            uint32_t offset,
                                            uint32_t scrambleConstant);
                                RNGSobolGPU(const RNGSobolGPU&) = delete;
        RNGSobolGPU&            operator=(const RNGSobolGPU&) = delete;
                                ~RNGSobolGPU() = default;

        __device__ float        Uniform() override;
        __device__ float        Uniform(float min, float max) override;
        __device__ float        Normal() override;
        __device__ float        Normal(float mean, float stdDev) override;
};

class RNGSobolCPU : public RNGeneratorCPUI
{
    private:
        DeviceMemory                                memRandom;
        std::map<const CudaGPU*, RNGeneratorGPUI**> deviceGenerators;

    public:
        // Constructors & Destructor
                            RNGSobolCPU() = default;
                            RNGSobolCPU(uint32_t seed,
                                        const CudaSystem&);
                            RNGSobolCPU(uint32_t seed,
                                        const CudaGPU&);
                            RNGSobolCPU(const RNGSobolCPU&) = delete;
                            RNGSobolCPU(RNGSobolCPU&&) = default;
        RNGSobolCPU&        operator=(const RNGSobolCPU&) = delete;
        RNGSobolCPU&        operator=(RNGSobolCPU&&) = default;
                            ~RNGSobolCPU() = default;

        RNGeneratorGPUI**   GetGPUGenerators(const CudaGPU&) override;
        size_t              UsedGPUMemory() const override;
};

inline size_t RNGSobolCPU::UsedGPUMemory() const
{
    return memRandom.Size();
}


__device__ __forceinline__
RNGSobolGPU::RNGSobolGPU(curandDirectionVectors32_t directionVectors,
                         uint32_t offset,
                         uint32_t scrambleConstant)
{
    curand_init(directionVectors,
                offset,
                scrambleConstant,
                &rState);
}


__device__ __forceinline__
float RNGSobolGPU::Uniform()
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

__device__ __forceinline__
float RNGSobolGPU::Uniform(float min, float max)
{
    float result = Uniform() * (min - max) + min;
    return result;
}

__device__ __forceinline__
float RNGSobolGPU::Normal()
{
    return curand_normal(&rState);
}

__device__ __forceinline__
float RNGSobolGPU::Normal(float mean, float stdDev)
{
    return curand_normal(&rState) * stdDev + mean;
}