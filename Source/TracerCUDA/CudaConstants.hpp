#pragma once

template<class Function, class... Args>
__host__
void CudaGPU::KC_X(uint32_t sharedMemSize,
                   cudaStream_t stream,
                   size_t workCount,
                   //
                   Function&& f, Args&&... args) const
{
    CUDA_CHECK(cudaSetDevice(deviceId));
    uint32_t blockCount = static_cast<uint32_t>((workCount + (StaticThreadPerBlock1D - 1)) / StaticThreadPerBlock1D);
    uint32_t blockSize = StaticThreadPerBlock1D;

    f<<<blockCount, blockSize, sharedMemSize, stream>>>(args...);
    //void* argsList[] = {((void*)&args)...};
    //CUDA_CHECK(cudaLaunchKernel(f, blockCount, blockSize, argsList,
    //                            sharedMemSize, stream));
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__
void CudaGPU::KC_XY(uint32_t sharedMemSize,
                    cudaStream_t stream,
                    size_t workCount,
                    //
                    Function&& f, Args&&... args) const
{
    CUDA_CHECK(cudaSetDevice(deviceId));
    size_t linearThreadCount = StaticThreadPerBlock2D[0] * StaticThreadPerBlock2D[1];
    size_t blockCount = (workCount + (linearThreadCount - 1)) / StaticThreadPerBlock1D;
    uint32_t blockSize = StaticThreadPerBlock1D;

    f<<<blockCount, blockSize, sharedMemSize, stream>>>(args...);
    //void* argsList[] = {((void*)&args)...};
    //CUDA_CHECK(cudaLaunchKernel(f, blockCount, blockSize, argsList,
    //                            sharedMemSize, stream));
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__
inline void CudaGPU::GridStrideKC_X(uint32_t sharedMemSize,
                                    cudaStream_t stream,
                                    size_t workCount,
                                    //
                                    Function&& f, Args&&... args) const
{
    const size_t threadCount = StaticThreadPerBlock1D;
    uint32_t blockCount = DetermineGridStrideBlock(sharedMemSize,
                                                   threadCount, workCount, &f);
    // Full potential GPU Call
    CUDA_CHECK(cudaSetDevice(deviceId));
    uint32_t blockSize = StaticThreadPerBlock1D;

    f<<<blockCount, blockSize, sharedMemSize, stream>>>(args...);
    //void* argsList[] = {((void*)&args)...};
    //CUDA_CHECK(cudaLaunchKernel(f, blockCount, blockSize, argsList,
    //                            sharedMemSize, stream));
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__
inline void CudaGPU::GridStrideKC_XY(uint32_t sharedMemSize,
                                     cudaStream_t stream,
                                     size_t workCount,
                                     //
                                     Function&& f, Args&&... args) const
{
    const size_t threadCount = StaticThreadPerBlock2D[0] * StaticThreadPerBlock2D[1];
    uint32_t blockCount = DetermineGridStrideBlock(sharedMemSize,
                                                   threadCount, workCount, &f);

    CUDA_CHECK(cudaSetDevice(deviceId));
    dim3 blockSize = dim3(StaticThreadPerBlock2D[0], StaticThreadPerBlock2D[1]);

    f<<<blockCount, blockSize, sharedMemSize, stream>>>(args...);
    //void* argsList[] = {((void*)&args)...};
    //CUDA_CHECK(cudaLaunchKernel(f, blockCount, blockSize, argsList,
    //                            sharedMemSize, stream));
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__ void CudaGPU::AsyncGridStrideKC_X(uint32_t sharedMemSize,
                                           size_t workCount,
                                           //
                                           Function&& f, Args&&... args) const
{
    const size_t threadCount = StaticThreadPerBlock1D;
    uint32_t requiredSMCount = DetermineGridStrideBlock(sharedMemSize,
                                                        threadCount, workCount, &f);
    cudaStream_t stream = DetermineStream(requiredSMCount);

    CUDA_CHECK(cudaSetDevice(deviceId));
    uint32_t blockSize = StaticThreadPerBlock1D;

    f<<<requiredSMCount, blockSize, sharedMemSize, stream>>>(args...);
    //void* argsList[] = {((void*)&args)...};
    //CUDA_CHECK(cudaLaunchKernel(f, requiredSMCount, blockSize, argsList,
    //                            sharedMemSize, stream));
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__ void CudaGPU::AsyncGridStrideKC_XY(uint32_t sharedMemSize,
                                            size_t workCount,
                                            //
                                            Function&& f, Args&&... args) const
{
    const size_t threadCount = StaticThreadPerBlock2D[0] * StaticThreadPerBlock2D[1];
    uint32_t requiredSMCount = DetermineGridStrideBlock(sharedMemSize,
                                                        threadCount, workCount, &f);
    cudaStream_t stream = DetermineStream(requiredSMCount);

    CUDA_CHECK(cudaSetDevice(deviceId));
    dim3 blockSize = dim3(StaticThreadPerBlock2D[0], StaticThreadPerBlock2D[1]);

    f<<<requiredSMCount, blockSize, sharedMemSize, stream>>>(args...);
    //void* argsList[] = {((void*)&args)...};
    //CUDA_CHECK(cudaLaunchKernel(f, requiredSMCount, blockSize, argsList,
    //                            sharedMemSize, stream));
    CUDA_KERNEL_CHECK();
}