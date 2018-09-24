#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/CudaConstants.h"
#include "RayLib/DeviceMemory.h"
#include "RayLib/Log.h"
#include "RayLib/CameraKernels.cuh"
#include "RayLib/TracerError.h"

TEST(CameraRayGPU, Test)
{
	TracerError e = CudaSystem::Initialize();

	static constexpr size_t ThreadCount = 32;
	static constexpr size_t StateSize = 32 * sizeof(uint32_t);

	static constexpr size_t NumberPerThread = 2;
	static constexpr size_t NumberCount = NumberPerThread * ThreadCount;
	static constexpr size_t NumberSize = NumberCount * sizeof(uint32_t);

	DeviceMemory randomState(StateSize);
	DeviceMemory numbers(NumberSize);

	//// Set State
	//std::mt19937 engine;
	//uint32_t* seeds = static_cast<uint32_t*>(randomState);
	//for(size_t i = 0; i < ThreadCount; i++)
	//{
	//	seeds[i] = engine();
	//}

	// Kernel Call
	uint32_t* h_data = static_cast<uint32_t*>(numbers);

	//KCGenerateCameraRays();


	CUDA_KERNEL_CHECK();
	CUDA_CHECK(cudaDeviceSynchronize());

	//for(int i = 0; i < NumberCount; i++)
	//{
	//	METU_LOG("%u", h_data[i]);
	//}
}