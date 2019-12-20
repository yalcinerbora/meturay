#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <random>

#include "TracerLib/Random.cuh"
#include "TracerLib/DeviceMemory.h"
#include "RayLib/Log.h"

__global__ void KInitRandStates(uint32_t seed,
                                curandStateMRG32k3a_t* states)
{
    curand_init(seed, threadIdx.x, 0, states);
}

__global__ void KRandomNumbers(RNGGMem gMemory,
                               uint32_t* randomNumbers,
                               size_t numberPerThread)
{
    extern __shared__ uint32_t sStates[];

    RandomGPU rand(gMemory.state);

    for(int i = 0; i < numberPerThread; i++)
    {
        int loc = i * blockDim.x + threadIdx.x;

        uint32_t r = rand.Generate();
        randomNumbers[loc] = r;
    }
}

TEST(RandomGPU, All)
{
    static uint32_t ThreadCount = 32;
    static size_t StateSize = ThreadCount * sizeof(curandStateMRG32k3a_t);

    static size_t NumberPerThread = 2;
    static size_t NumberCount = NumberPerThread * ThreadCount;
    static size_t NumberSize = NumberCount * sizeof(uint32_t);

    DeviceMemory randomState(StateSize);
    DeviceMemory numbers(NumberSize);

    // Set State
    std::mt19937 engine(2109);
    curandStateMRG32k3a_t* states = static_cast<curandStateMRG32k3a_t*>(randomState);
   
    KInitRandStates<<<1, ThreadCount, StateSize>>>(engine(), states);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel Call
    uint32_t* h_data = static_cast<uint32_t*>(numbers);
    KRandomNumbers<<<1, ThreadCount, StateSize>>>({states}, h_data,
                                                  NumberPerThread);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    // TODO: How to check this??

    //for(int i = 0; i < NumberCount; i++)
    //{
    //    METU_LOG("%u", h_data[i]);
    //}
}