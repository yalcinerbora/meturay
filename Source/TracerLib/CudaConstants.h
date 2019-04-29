#pragma once

/**

Compile Time Cuda Constants

Thread per Block etc..

*/

#ifdef METU_SHARED_GPULIST
#define METU_SHARED_TRACER_ENTRY_POINT __declspec(dllexport)
#else
#define METU_SHARED_TRACER_ENTRY_POINT __declspec(dllimport)
#endif

#include <cuda.h>
#include <vector>
#include <array>

#include "RayLib/Vector.h"

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
			GPU_TURING_VOLTA
		};

		static GPUTier				DetermineGPUTier(cudaDeviceProp);

	private:
		template<int Count>
		struct WorkGroup
		{
			std::array<cudaStream_t, Count>	works = {};
			mutable int						currentIndex = 0;

			// Constructors
											WorkGroup();
											WorkGroup(const WorkGroup&) = delete;
											WorkGroup(WorkGroup&&);
			WorkGroup&						operator=(const WorkGroup&) = delete;
			WorkGroup&						operator=(WorkGroup&&);
											~WorkGroup();

			cudaStream_t					UseGroup() const;
		};

		int							deviceId;
		cudaDeviceProp				props;
		// Generated Data
		GPUTier						tier;
		//
		WorkGroup<8>				smallWorkList;
		WorkGroup<4>				mediumWorkList;
		WorkGroup<2>				largeWorkList;

	protected:
	public:
		// Constrctors & Destructor
								CudaGPU(int deviceId);
								CudaGPU(const CudaGPU&) = delete;
								CudaGPU(CudaGPU&&) = default;
		CudaGPU&				operator=(const CudaGPU&) = delete;
		CudaGPU&				operator=(CudaGPU&&) = default;
								~CudaGPU() = default;
		//
		int						DeviceId() const;
		std::string				Name() const;
		double					TotalMemoryMB() const;
		double					TotalMemoryGB() const;
		GPUTier					Tier() const;

		size_t					TotalMemory() const;
		Vector2i				MaxTexture2DSize() const;

		uint32_t				SMCount() const;
		uint32_t				RecommendedBlockCountPerSM(void* kernkernelFuncelPtr,
														   uint32_t threadsPerBlock = StaticThreadPerBlock1D,
														   uint32_t sharedMemSize = 0) const;
		cudaStream_t			DetermineStream(uint32_t requiredSMCount) const;
};

class CudaSystem
{
	public:
		enum CudaError
		{
			CUDA_SYSTEM_UNINIALIZED,
			// Initalization Errors
			OLD_DRIVER,
			NO_DEVICE,
			// Ok
			OK
		};

	private:
		static std::vector<CudaGPU>			systemGPUs;
		static CudaError					systemStatus;

		static uint32_t						DetermineGridStrideBlock(int gpuIndex,
																	 uint32_t sharedMemSize,
																	 uint32_t threadCount,
																	 size_t workCount,
																	 void* func);
	protected:
	public:
		// Constructors & Destructor
											CudaSystem() = delete;
											CudaSystem(const CudaSystem&) = delete;

		static CudaError					Initialize(std::vector<CudaGPU>& gpus);
		static CudaError					Initialize();

		static CudaError					SystemStatus();



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
		// Only for grid strided kernels, and for specific GPU
		// Material calls require difrrent GPUs (texture sharing)
		// TODO:
		template<class Function, class... Args>
		static __host__ void						AsyncGridStrideKC_X(int gpuIndex,
																		uint32_t sharedMemSize,
																		size_t workCount,
																		//
																		Function&&, Args&&...);

		template<class Function, class... Args>
		static __host__ void						AsyncGridStrideKC_XY(int gpuIndex,
																		 uint32_t sharedMemSize,
																		 size_t workCount,
																		 //
																		 Function&&, Args&&...);

		// Multi-Device Splittable Smart GPU Calls
		// Automatic device split and stream split on devices
		static const std::vector<size_t>			GridStrideMultiGPUSplit(size_t workCount,
																			uint32_t threadCount,
																			uint32_t sharedMemSize,
																			void* f);

		// Misc
		static const std::vector<CudaGPU>&			GPUList();
		static bool									SingleGPUSystem();
		static void									SyncAllGPUs();

		static constexpr int						CURRENT_DEVICE = -1;
};

// Verbosity
using SystemGPUs = std::vector<CudaGPU>;

template<int Count>
CudaGPU::WorkGroup<Count>::WorkGroup()
	: currentIndex(0)
{
	for(int i = 0; i < Count; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&works[i]));
	}
}

template<int Count>
CudaGPU::WorkGroup<Count>::WorkGroup(WorkGroup&& other)
	: currentIndex(0)
{
	for(int i = 0; i < Count; i++)
	{
		works[i] = other.works[i];
		other.works[i] = nullptr;
	}
}

template<int Count>
CudaGPU::WorkGroup<Count>& CudaGPU::WorkGroup<Count>::operator=(WorkGroup&& other)
{
	assert(this != &other);
	currentIndex = other.currentIndex;
	for(int i = 0; i < Count; i++)
	{
		works[i] = other.works[i];
		other.works[i] = nullptr;
	}
	return *this;
}

template<int Count>
CudaGPU::WorkGroup<Count>::~WorkGroup()
{
	for(int i = 0; i < Count; i++)
	{
		if(works[i]) CUDA_CHECK(cudaStreamDestroy(works[i]));
	}
}

template<int Count>
cudaStream_t CudaGPU::WorkGroup<Count>::UseGroup() const
{
	int i = currentIndex;
	currentIndex = (currentIndex + 1) % Count;
	return works[i];
}

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
	const size_t threadCount = StaticThreadPerBlock1D;
	uint32_t blockCount = DetermineGridStrideBlock(gpuIndex, sharedMemSize,
												   threadCount, workCount, &f);

	// Full potential GPU Call
	CUDA_CHECK(cudaSetDevice(gpuIndex));
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
	const size_t threadCount = StaticThreadPerBlock2D[0] * StaticThreadPerBlock2D[1];
	uint32_t blockCount = DetermineGridStrideBlock(gpuIndex, sharedMemSize,
												   threadCount, workCount, &f);

	CUDA_CHECK(cudaSetDevice(gpuIndex));
	dim3 blockSize = dim3(StaticThreadPerBlock2D[0], StaticThreadPerBlock2D[1]);
	f<<<blockCount, blockSize, sharedMemSize, stream>>>(args...);
	CUDA_KERNEL_CHECK();
}


template<class Function, class... Args>
__host__ void CudaSystem::AsyncGridStrideKC_X(int gpuIndex,
											  uint32_t sharedMemSize,
											  size_t workCount,
											  //
											  Function&& f, Args&&... args)
{
	const size_t threadCount = StaticThreadPerBlock1D;
	uint32_t requiredSMCount = DetermineGridStrideBlock(gpuIndex, sharedMemSize,
														threadCount, workCount, &f);
	cudaStream_t stream = GPUList()[gpuIndex].DetermineStream(requiredSMCount);

	CUDA_CHECK(cudaSetDevice(gpuIndex));
	uint32_t blockSize = StaticThreadPerBlock1D;
	f<<<requiredSMCount, blockSize, sharedMemSize, stream>>>(args...);
	CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__ void CudaSystem::AsyncGridStrideKC_XY(int gpuIndex,
											   uint32_t sharedMemSize,
											   size_t workCount,
											   //
											   Function&& f, Args&&... args)
{
	const size_t threadCount = StaticThreadPerBlock2D[0] * StaticThreadPerBlock2D[1];
	uint32_t requiredSMCount = DetermineGridStrideBlock(gpuIndex, sharedMemSize,
														thread, workCount, &f);
	cudaStream_t stream = GPUList()[gpuIndex].DetermineStream(requiredSMCount);

	CUDA_CHECK(cudaSetDevice(gpuIndex));
	dim3 blockSize = dim3(StaticThreadPerBlock2D[0], StaticThreadPerBlock2D[1]);
	f<<<requiredSMCount, blockSize, sharedMemSize, stream>>>(args...);
	CUDA_KERNEL_CHECK();
}

inline const std::vector<CudaGPU>& CudaSystem::GPUList()
{
	if(systemGPUs.size() == 0)
		systemStatus = Initialize();
	return systemGPUs;
}

inline bool CudaSystem::SingleGPUSystem()
{
	return GPUList().size() == 1;
}

inline void CudaSystem::SyncAllGPUs()
{
	int currentDevice;
	CUDA_CHECK(cudaGetDevice(&currentDevice));

	for(int i = 0; i < static_cast<int>(GPUList().size()); i++)
	{
		CUDA_CHECK(cudaSetDevice(i));
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	CUDA_CHECK(cudaSetDevice(currentDevice));
}