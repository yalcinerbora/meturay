#pragma once

/**

Compile Time Cuda Constants

Thread per Block etc..

*/

#include <cuda.h>
#include <vector>
#include "Vector.h"

// Thread Per Block
static constexpr int BlockPerSM = 16;
static constexpr int StaticThreadPerBlock1D = 256;					
static constexpr Vector2i StaticThreadPerBlock2D = Vector2i(16, 16);

class CudaGPU
{
	public:
		enum GPUTier
		{
			UNSUPPORTED,
			KEPLER,
			MAXWELL,
			PASCAL
		};

		static GPUTier			DetermineGPUTier(cudaDeviceProp);

	private:
		int						deviceId;
		cudaDeviceProp			props;

		// Generated Data
		int32_t					gridStrideBlockCount;
		GPUTier					tier;

	protected:
	public:
		// Constrctors & Destructor
								CudaGPU(int deviceId);
								~CudaGPU() = default;
		//
		int						DeviceId() const;
		std::string				Name() const;
		double					TotalMemoryMB() const;
		double					TotalMemoryGB() const;

		size_t					TotalMemory() const;
		Vector2i				MaxTexture2DSize() const;
		int32_t					RecommendedBlockCount() const;

};

class CudaSystem
{
	private:
		static std::vector<CudaGPU>	gpus;

	protected:
	public:
		// Constructors & Destructor
									CudaSystem() = delete;
									CudaSystem(const CudaSystem&) = delete;

		//
		static bool					Initialize();

		// Convenience Functions For Kernel Call
		template<class Function, class... Args>
		__host__ void				GPUCallX(int deviceId,
											 cudaStream_t stream,
											 size_t sharedMemSize,
											 Function&& f, Args&&...);
		template<class Function, class... Args>
		__host__ void				GPUCallXY(int deviceId,
											 cudaStream_t stream,
											 size_t sharedMemSize,
											 Function&& f, Args&&...);
};

template<class Function, class... Args>
__host__
inline void CudaSystem::GPUCallX(int deviceId, 
								 cudaStream_t stream,
								 size_t sharedMemSize,
								 Function&& f, Args&&... args)
{
	const CudaGPU& gpu = gpus[deviceId];

	CUDA_CHECK(cudaSetDevice(deviceId));
	uint32_t blockCount = gpu.RecommendedBlockCount();
	uint32_t blockSize = StaticThreadPerBlock1D;
	f<<<blockCount, blockSize, sharedMemsize, stream>>>(args...);
	CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__
inline void CudaSystem::GPUCallXY(int deviceId, 
								  cudaStream_t stream,
								 size_t sharedMemSize,
								 Function&& f, Args&&... args)
{
	const CudaGPU& gpu = gpus[deviceId];

	CUDA_CHECK(cudaSetDevice(deviceId));
	uint32_t blockCount = gpu.RecommendedBlockCount();
	dim3 blockSize = dim3(StaticThreadPerBlock2D[0], StaticThreadPerBlock2D[1]);
	f<<<blockCount, blockSize, sharedMemsize, stream>>>(args...);
	CUDA_KERNEL_CHECK();
}