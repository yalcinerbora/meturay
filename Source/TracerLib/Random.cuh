#pragma once

/**

GPU Linear Congruential Generator

Implementation of Warp Std Generator

*/

#include <limits>
#include <cuda_runtime.h>
#include "RayLib/Vector.h"
#include "CudaConstants.h"
#include "RNGStructs.h"

#define GLOBAL_ID_X (threadIdx.x + blockDim.x * blockIdx.x)
#define GLOBAL_ID_Y (threadIdx.y + blockDim.y * blockIdx.y)
#define LINEAR_GLOBAL_ID (GLOBAL_ID_X + GLOBAL_ID_Y * gridDim.x)
#define LINEAR_LOCAL_ID (threadIdx.x + threadIdx.y * blockDim.x)

class RandomGPU
{
    public:
        static constexpr uint32_t   Max = std::numeric_limits<uint32_t>::max();
        static constexpr uint32_t   Min = std::numeric_limits<uint32_t>::min();
        static constexpr uint32_t   Range = Max-Min;

    private:
        curandStateMRG32k3a_t*      gStates;
        curandStateMRG32k3a_t       rState;

    protected:
    public:
        // Constructor
        __device__                  RandomGPU(curandStateMRG32k3a_t* gStates);
                                    RandomGPU(const RandomGPU&) = delete;
        RandomGPU&                  operator=(const RandomGPU&) = delete;
        __device__                  ~RandomGPU();

        // Fundemental Generation Function
        __device__  uint32_t        Generate();

        __device__                  operator curandStateMRG32k3a_t*();
};

__device__
inline RandomGPU::RandomGPU(curandStateMRG32k3a_t* gStates)
    : gStates(gStates)
    , rState(gStates[LINEAR_GLOBAL_ID])
{}

__device__
inline RandomGPU::~RandomGPU()
{
    gStates[LINEAR_GLOBAL_ID] = rState;
}

__device__
inline uint32_t RandomGPU::Generate()
{
    return curand(&rState);
}

inline __device__ RandomGPU::operator curandStateMRG32k3a_t*()
{
    return &rState;
}

// Pseduo Uniform Generation
namespace GPUDistribution
{
    template <class T, typename = FloatEnable<T>>
    __device__ T Uniform(RandomGPU& r)
    {
        return static_cast<T>(1.0f - curand_uniform(r));
    }

    template <class T, typename = FloatEnable<T>>
    __device__ T Uniform(RandomGPU& r, T min, T max)
    {
        return static_cast<T>(1.0f - curand_uniform(r)) * (min - max) + min;
    }

    template <class T, typename = FloatEnable<T>>
    __device__ T Normal(RandomGPU& r);

    template <class T, typename = FloatEnable<T>>
    __device__ T Normal(RandomGPU& r, T mean, T stdDev);
}

template<>
__device__
inline float GPUDistribution::Normal(RandomGPU& r)
{
    return curand_normal(r);
}

template<>
__device__
inline double GPUDistribution::Normal(RandomGPU& r)
{
    return curand_normal_double(r);
}

template<>
__device__
inline float GPUDistribution::Normal(RandomGPU& r, float mean, float stdDev)
{
    return curand_normal(r) * stdDev + mean;
}

template<>
__device__
inline double GPUDistribution::Normal(RandomGPU& r, double mean, double stdDev)
{
    return curand_normal_double(r) * stdDev + mean;
}