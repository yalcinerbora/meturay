#pragma once

/**

Compile Time Cuda Constants

Thread per Block etc..

*/

#include <cuda.h>
#include <vector>
#include "Vector.h"

// Except first generation this did not change having this compile time constant is a bliss
static constexpr unsigned int WarpSize = 32;

// Thread Per Block Constants
static constexpr unsigned int BlockPerSM = 16;
static constexpr unsigned int StaticThreadPerBlock1D = 256;
static constexpr unsigned int StaticThreadPerBlock2D_X = 16;
static constexpr unsigned int StaticThreadPerBlock2D_Y = 16;
static constexpr Vector2ui StaticThreadPerBlock2D = Vector2ui(StaticThreadPerBlock2D_X,
															  StaticThreadPerBlock2D_Y);

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

		static bool							Initialize();

		// Convenience Functions For Kernel Call
		template<class Function, class... Args>
		static __host__ void						GPUCallX(int deviceId,
															 cudaStream_t stream,
															 size_t sharedMemSize,
															 Function&& f, Args&&...);
		template<class Function, class... Args>
		static __host__ void						GPUCallXY(int deviceId,
															  cudaStream_t stream,
															  size_t sharedMemSize,
															  Function&& f, Args&&...);

		static const std::vector<CudaGPU>	GPUList();


		static constexpr int				CURRENT_DEVICE = -1;
};

template<class Function, class... Args>
__host__
inline void CudaSystem::GPUCallX(int deviceId, 
								 cudaStream_t stream,
								 size_t sharedMemSize,
								 Function&& f, Args&&... args)
{
	if(deviceId != CURRENT_DEVICE)
	{
		CUDA_CHECK(cudaSetDevice(deviceId));
	}
	else
	{
		CUDA_CHECK(cudaGetDevice(&deviceId));
	}

	const CudaGPU& gpu = gpus[deviceId];
	uint32_t blockCount = gpu.RecommendedBlockCount();
	uint32_t blockSize = StaticThreadPerBlock1D;
	f<<<blockCount, blockSize, sharedMemSize, stream>>>(args...);
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

	if(deviceId != CURRENT_DEVICE)
	{
		CUDA_CHECK(cudaSetDevice(deviceId));
	}
	uint32_t blockCount = gpu.RecommendedBlockCount();
	dim3 blockSize = dim3(StaticThreadPerBlock2D[0], StaticThreadPerBlock2D[1]);
	f<<<blockCount, blockSize, sharedMemSize, stream>>>(args...);
	CUDA_KERNEL_CHECK();
}

inline const std::vector<CudaGPU> CudaSystem::GPUList()
{
	return gpus;
}