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
        curandStateSobol32_t    rStateX;
        curandStateSobol32_t    rStateY;
        curandStateSobol32_t    rStateMain;

        __device__ float        UniformInner(curandStateSobol32_t*);

    public:
        // Constructor
                                RNGSobolGPU() = default;
        __device__              RNGSobolGPU(curandDirectionVectors32_t,
                                            curandDirectionVectors32_t,
                                            curandDirectionVectors32_t,
                                            uint32_t offset);
                                RNGSobolGPU(const RNGSobolGPU&) = delete;
        RNGSobolGPU&            operator=(const RNGSobolGPU&) = default;
                                ~RNGSobolGPU() = default;

        __device__ float        Uniform() override;
        __device__ float        Uniform(float min, float max) override;
        __device__ Vector2f     Uniform2D() override;
        __device__ float        Normal() override;
        __device__ float        Normal(float mean, float stdDev) override;

        __device__ void         Skip(uint32_t skipCount);
};

class RNGScrSobolGPU final : public RNGeneratorGPUI
{
    private:
        curandStateScrambledSobol32_t rStateX;
        curandStateScrambledSobol32_t rStateY;
        curandStateScrambledSobol32_t rStateMain;

        __device__ float        UniformInner(curandStateScrambledSobol32_t*);

    public:
        // Constructor
                                RNGScrSobolGPU() = default;
        __device__              RNGScrSobolGPU(curandDirectionVectors32_t,
                                               curandDirectionVectors32_t,
                                               curandDirectionVectors32_t,
                                               uint32_t scrambleConstantX,
                                               uint32_t scrambleConstantY,
                                               uint32_t scrambleConstantMain,
                                               uint32_t offset);
                                RNGScrSobolGPU(const RNGScrSobolGPU&) = delete;
        RNGScrSobolGPU&         operator=(const RNGScrSobolGPU&) = default;
                                ~RNGScrSobolGPU() = default;

        __device__ float        Uniform() override;
        __device__ float        Uniform(float min, float max) override;
        __device__ Vector2f     Uniform2D() override;
        __device__ float        Normal() override;
        __device__ float        Normal(float mean, float stdDev) override;

        __device__ void         Skip(uint32_t skipCount);
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

        void                ExpandGenerator(DeviceMemory& genMemory,
                                            RNGeneratorGPUI**& dOffsetedGenerators,
                                            uint32_t generatorIndex,
                                            uint32_t generatorCount,
                                            uint32_t offsetPerGenerator,
                                            uint32_t extraOffsetThreshold,
                                            const CudaGPU&);
};

class RNGScrSobolCPU : public RNGeneratorCPUI
{
    private:
        DeviceMemory                                memRandom;
        std::map<const CudaGPU*, RNGeneratorGPUI**> deviceGenerators;

    public:
        // Constructors & Destructor
                            RNGScrSobolCPU() = default;
                            RNGScrSobolCPU(uint32_t seed,
                                           const CudaSystem&);
                            RNGScrSobolCPU(uint32_t seed,
                                           const CudaGPU&);
                            RNGScrSobolCPU(uint32_t seed,
                                           const CudaGPU&,
                                           uint32_t rngCount);
                            RNGScrSobolCPU(const RNGScrSobolCPU&) = delete;
                            RNGScrSobolCPU(RNGScrSobolCPU&&) = default;
        RNGScrSobolCPU&     operator=(const RNGScrSobolCPU&) = delete;
        RNGScrSobolCPU&     operator=(RNGScrSobolCPU&&) = default;
                            ~RNGScrSobolCPU() = default;

        RNGeneratorGPUI**   GetGPUGenerators(const CudaGPU&) override;
        size_t              UsedGPUMemory() const override;

        void                ExpandGenerator(DeviceMemory& genMemory,
                                            RNGeneratorGPUI**& dOffsetedGenerators,
                                            uint32_t generatorIndex,
                                            uint32_t generatorCount,
                                            uint32_t offsetPerGenerator,
                                            uint32_t extraOffsetThreshold,
                                            const CudaGPU&);
};

inline size_t RNGSobolCPU::UsedGPUMemory() const
{
    return memRandom.Size();
}

inline size_t RNGScrSobolCPU::UsedGPUMemory() const
{
    return memRandom.Size();
}

__device__ inline
RNGSobolGPU::RNGSobolGPU(curandDirectionVectors32_t dirVecX,
                         curandDirectionVectors32_t dirVecY,
                         curandDirectionVectors32_t dirVecMain,
                         uint32_t offset)
{
    curand_init(dirVecX, offset, &rStateX);
    curand_init(dirVecY, offset, &rStateY);
    curand_init(dirVecMain, offset, &rStateMain);
}

__device__ inline
float RNGSobolGPU::UniformInner(curandStateSobol32_t* state)
{
    // curand returns (0, 1]
    // we need [0, 1) invert it
    float result = 1 - curand_uniform(state);
    // TODO: curand rarely returns 1.0f anyways
    // (is it a bug?, or "1.0f - x" is causing the bug)
    // Just if it returns 1.0f return nearest float closest to 1.0f
    result = (result == 1) ? nextafter(result, 0.0f) : result;
    return result;
}

__device__ inline
float RNGSobolGPU::Uniform()
{
    return UniformInner(&rStateMain);
}

__device__ inline
Vector2f RNGSobolGPU::Uniform2D()
{
    return Vector2f(UniformInner(&rStateX),
                    UniformInner(&rStateY));
}

__device__ inline
float RNGSobolGPU::Uniform(float min, float max)
{
    float result = Uniform() * (min - max) + min;
    return result;
}

__device__ inline
float RNGSobolGPU::Normal()
{
    return curand_normal(&rStateMain);
}

__device__ inline
float RNGSobolGPU::Normal(float mean, float stdDev)
{
    return curand_normal(&rStateMain) * stdDev + mean;
}

__device__ inline
void RNGSobolGPU::Skip(uint32_t skipCount)
{
    skipahead<curandStateSobol32_t*>(skipCount, &rStateX);
    skipahead<curandStateSobol32_t*>(skipCount, &rStateY);
    skipahead<curandStateSobol32_t*>(skipCount, &rStateMain);
}

__device__ inline
RNGScrSobolGPU::RNGScrSobolGPU(curandDirectionVectors32_t dirVecX,
                               curandDirectionVectors32_t dirVecY,
                               curandDirectionVectors32_t dirVecMain,
                               uint32_t scrambleConstantX,
                               uint32_t scrambleConstantY,
                               uint32_t scrambleConstantMain,
                               uint32_t offset)
{
    curand_init(dirVecX, scrambleConstantX,
                offset, &rStateX);

    curand_init(dirVecY, scrambleConstantY,
                offset, &rStateY);

    curand_init(dirVecMain, scrambleConstantMain,
                offset, &rStateMain);
}

__device__ inline
float RNGScrSobolGPU::UniformInner(curandStateScrambledSobol32_t* state)
{
    // curand returns (0, 1]
    // we need [0, 1) invert it
    float result = 1 - curand_uniform(state);
    // TODO: curand rarely returns 1.0f anyways
    // (is it a bug?, or "1.0f - x" is causing the bug)
    // Just if it returns 1.0f return nearest float closest to 1.0f
    result = (result == 1) ? nextafter(result, 0.0f) : result;
    return result;
}

__device__ inline
float RNGScrSobolGPU::Uniform()
{
    return UniformInner(&rStateMain);
}

__device__ inline
Vector2f RNGScrSobolGPU::Uniform2D()
{
    return Vector2f(UniformInner(&rStateX),
                    UniformInner(&rStateY));
}

__device__ inline
float RNGScrSobolGPU::Uniform(float min, float max)
{
    float result = Uniform() * (min - max) + min;
    return result;
}

__device__ inline
float RNGScrSobolGPU::Normal()
{
    return curand_normal(&rStateMain);
}

__device__ inline
float RNGScrSobolGPU::Normal(float mean, float stdDev)
{
    return curand_normal(&rStateMain) * stdDev + mean;
}

__device__ inline
void RNGScrSobolGPU::Skip(uint32_t skipCount)
{
    skipahead<curandStateScrambledSobol32_t*>(skipCount, &rStateX);
    skipahead<curandStateScrambledSobol32_t*>(skipCount, &rStateY);
    skipahead<curandStateScrambledSobol32_t*>(skipCount, &rStateMain);
}