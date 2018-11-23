#pragma once

/**

Compile Time Cuda Constants

Thread per Block etc..

*/

#include <cuda.h>
#include <vector>

#include "RayLib/Vector.h"

struct TracerError;

// Except first generation this did not change having this compile time constant is a bliss
static constexpr unsigned int WarpSize = 32;

// Thread Per Block Constants
static constexpr unsigned int BlocksPerSM = 32;
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
			GPU_UNSUPPORTED,
			GPU_KEPLER,
			GPU_MAXWELL,
			GPU_PASCAL,			
			GPU_TURING
		};

		static GPUTier			DetermineGPUTier(cudaDeviceProp);

	private:
		int						deviceId;
		cudaDeviceProp			props;
		// Generated Data
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

		uint32_t				SMCount() const;
		uint32_t				RecommendedBlockCountPerSM(void* kernkernelFuncelPtr,
														   uint32_t threadsPerBlock = StaticThreadPerBlock1D,
														   uint32_t sharedMemSize = 0) const;
};

class CudaSystem
{
	private:
		static std::vector<CudaGPU>			systemGPUs;

	protected:
	public:
		// Constructors & Destructor
											CudaSystem() = delete;
											CudaSystem(const CudaSystem&) = delete;

		static TracerError					Initialize(std::vector<CudaGPU>& gpus);
		static TracerError					Initialize();

		// Classic GPU Calls
		// Create just enough blocks according to work size
		template<class Function, class... Args>
		static __host__ void						KC_X(int gpuIndex,
														 uint32_t sharedMemSize,
														 cudaStream_t stream,
														 size_t workCount,
														 //
														 Function&& f, Args&&...);
		template<class Function, class... Args>
		static __host__ void						KC_XY(int gpuIndex,
														  uint32_t sharedMemSize,
														  cudaStream_t stream,
														  size_t workCount,
														  //
														  Function&& f, Args&&...);
		
		// Grid-Stride Kernels
		// Convenience Functions For Kernel Call
		// Simple full GPU utilizing calls over a stream		
		template<class Function, class... Args>
		static __host__ void						GridStrideKC_X(int gpuIndex,
																   uint32_t sharedMemSize,
																   cudaStream_t stream,
																   size_t workCount,
																   //
																   Function&&, Args&&...);

		template<class Function, class... Args>
		static __host__ void						GridStrideKC_XY(int gpuIndex,
																	uint32_t sharedMemSize,
																	cudaStream_t stream,
																	size_t workCount,
																	//
																	Function&&, Args&&...);

		// Smart GPU Calls
		// Automatic stream split
		// TODO:

		// Multi-Device Splittable Smart GPU Calls
		// Automatic device split and stream split
		//TODO:

		// Misc
		static const std::vector<CudaGPU>			GPUList();
		static bool									SingleGPUSystem();
		static void									SyncAllGPUs();

		static constexpr int						CURRENT_DEVICE = -1;
};

template<class Function, class... Args>
__host__ 
void CudaSystem::KC_X(int gpuIndex,
					  uint32_t sharedMemSize,
					  cudaStream_t stream,
					  size_t workCount,
					  //
					  Function&& f, Args&&... args)
{
	CUDA_CHECK(cudaSetDevice(gpuIndex));
	uint32_t blockCount = static_cast<uint32_t>((workCount + (StaticThreadPerBlock1D - 1)) / StaticThreadPerBlock1D);
	uint32_t blockSize = StaticThreadPerBlock1D;
	f<<<blockCount, blockSize, sharedMemSize, stream>>>(args...);
	CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__ 
void CudaSystem::KC_XY(int gpuIndex,
					   uint32_t sharedMemSize,
					   cudaStream_t stream,
					   size_t workCount,
					   //
					   Function&& f, Args&&... args)
{
	CUDA_CHECK(cudaSetDevice(gpuIndex));
	size_t linearThreadCount = StaticThreadPerBlock2D[0] * StaticThreadPerBlock2D[1];
	size_t blockCount = (workCount + (linearThreadCount - 1)) / StaticThreadPerBlock1D;
	uint32_t blockSize = StaticThreadPerBlock1D;
	f<<<blockCount, blockSize, sharedMemSize, stream>>> (args...);
	CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__
inline void CudaSystem::GridStrideKC_X(int gpuIndex,									   
									   uint32_t sharedMemSize,
									   cudaStream_t stream,									   
									   size_t workCount,
									   //
									   Function&& f, Args&&... args)
{
	CUDA_CHECK(cudaSetDevice(gpuIndex));
	const CudaGPU& gpu = systemGPUs[gpuIndex];
	const size_t threadCount = StaticThreadPerBlock1D;
	uint32_t blockPerSM = gpu.RecommendedBlockCountPerSM(&f, StaticThreadPerBlock1D,
														 sharedMemSize);
	// Only call enough SM
	uint32_t totalRequiredBlocks = static_cast<uint32_t>((workCount + (threadCount - 1)) / threadCount);
	uint32_t requiredSMCount = totalRequiredBlocks / blockPerSM;
	uint32_t smCount = std::min(gpu.SMCount(), requiredSMCount);
	uint32_t blockCount = smCount * blockPerSM;

	// Full potential GPU Call
	uint32_t blockSize = StaticThreadPerBlock1D;
	f<<<blockCount, blockSize, sharedMemSize, stream>>> (args...);
	CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__
inline void CudaSystem::GridStrideKC_XY(int gpuIndex,									
										uint32_t sharedMemSize,										
										cudaStream_t stream,
										size_t workCount,
										//
										Function&& f, Args&&... args)
{
	CUDA_CHECK(cudaSetDevice(gpuIndex));
	const CudaGPU& gpu = systemGPUs[gpuIndex];
	const size_t threadCount = StaticThreadPerBlock2D[0] * StaticThreadPerBlock2D[1];
	uint32_t blockPerSM = gpu.RecommendedBlockCountPerSM(&f, threadCount,
														 sharedMemSize);
	// Only call enough SM
	uint32_t totalRequiredBlocks = static_cast<uint32_t>((workCount + (threadCount - 1)) / threadCount);
	uint32_t requiredSMCount = totalRequiredBlocks / blockPerSM;
	uint32_t smCount = std::min(gpu.SMCount(), requiredSMCount);
	uint32_t blockCount = smCount * blockPerSM;
	
	dim3 blockSize = dim3(StaticThreadPerBlock2D[0], StaticThreadPerBlock2D[1]);
	f<<<blockCount, blockSize, sharedMemSize, stream>>>(args...);
	CUDA_KERNEL_CHECK();
}

inline const std::vector<CudaGPU> CudaSystem::GPUList()
{
	return systemGPUs;
}

inline bool CudaSystem::SingleGPUSystem()
{
	return systemGPUs.size() == 1;
}

inline void CudaSystem::SyncAllGPUs()
{
	int currentDevice;
	CUDA_CHECK(cudaGetDevice(&currentDevice));

	for(int i = 0; i < static_cast<int>(systemGPUs.size()); i++)
	{
		CUDA_CHECK(cudaSetDevice(i));
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	CUDA_CHECK(cudaSetDevice(currentDevice));
}