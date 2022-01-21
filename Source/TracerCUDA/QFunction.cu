#include "QFunction.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"
#include "ParallelMemset.cuh"

#include <execution>

void QFunctionCPU::RecalculateDistributions(const CudaSystem& system)
{
    std::vector<const float*> dFuncPtrs;
    dFuncPtrs.resize(spatialCount, nullptr);
    for(uint32_t i = 0; i < spatialCount; i++)
    {
        dFuncPtrs[i] = qFuncGPU.gQFunction + i * qFuncGPU.dataPerNode.Multiply();
    }
    distributions.UpdateDistributions(dFuncPtrs, std::vector<bool>(spatialCount, true),
                                      system, cudaMemcpyDeviceToDevice);
}

TracerError QFunctionCPU::Initialize(const CudaSystem& system)
{
    // Initially set all values to uniform
    const CudaGPU& gpu = system.BestGPU();

    uint32_t dataCount = spatialCount * qFuncGPU.dataPerNode.Multiply();
    gpu.GridStrideKC_X(0, (cudaStream_t)0, dataCount,
                       //
                       KCMemset<float>,
                       //
                       qFuncGPU.gQFunction,
                       1.0f,
                       dataCount);

    // Generate Ptrs
    std::vector<const float*> dFuncPtrs(spatialCount, nullptr);
    for(uint32_t i = 0; i < spatialCount; i++)
    {
        dFuncPtrs[i] = qFuncGPU.gQFunction + i * qFuncGPU.dataPerNode.Multiply();
    }

    // Generate Distributions over this
    distributions = CPUDistGroupPiecewiseConst2D(dFuncPtrs,
                                                 std::vector<Vector2ui>(spatialCount, qFuncGPU.dataPerNode),
                                                 std::vector<bool>(spatialCount, true),
                                                 system);

    CUDA_CHECK(cudaMemcpy(const_cast<GPUDistPiecewiseConst2D*>(qFuncGPU.gDistributions),
                          distributions.DistributionGPU().data(),
                          sizeof(GPUDistPiecewiseConst2D) * spatialCount,
                          cudaMemcpyHostToDevice));

    return TracerError::OK;
}