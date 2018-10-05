#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <random>

#include "TracerLib/Random.cuh"
#include "RayLib/DeviceMemory.h"
#include "RayLib/Log.h"

__global__ void KRandomNumbers(RNGGMem gMemory,
							   uint32_t* randomNumbers,
							   size_t numberPerThread)
{
	extern __shared__ uint32_t sStates[];

	RandomGPU rand(gMemory.state, sStates);

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
	static size_t StateSize = ThreadCount * sizeof(uint32_t);

	static size_t NumberPerThread = 2;
	static size_t NumberCount = NumberPerThread * ThreadCount;
	static size_t NumberSize = NumberCount * sizeof(uint32_t);

	DeviceMemory randomState(StateSize);
	DeviceMemory numbers(NumberSize);
	
	// Set State
	std::mt19937 engine(2109);
	uint32_t* seeds = static_cast<uint32_t*>(randomState);
	for(size_t i = 0; i < ThreadCount; i++)
	{
		seeds[i] = engine();
	}

	// Kernel Call
	uint32_t* h_data = static_cast<uint32_t*>(numbers);
	KRandomNumbers<<<1, ThreadCount, StateSize>>>({seeds}, h_data,
												  NumberPerThread);
	CUDA_KERNEL_CHECK();
	CUDA_CHECK(cudaDeviceSynchronize());

	
	for(int i = 0; i < NumberCount; i++)
	{
		METU_LOG("%u", h_data[i]);
	}
}