#include "QFunction.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"
#include "ParallelMemset.cuh"

void QFunctionCPU::RecalculateDistributions(const CudaSystem&)
{

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

    // Generate Distributions over this
    //GPUP


    return TracerError::OK;
}