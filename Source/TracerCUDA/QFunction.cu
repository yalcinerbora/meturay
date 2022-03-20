#include "QFunction.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"
#include "ParallelMemset.cuh"

#include <execution>

#include "TracerDebug.h"

void QFunctionCPU::RecalculateDistributions(const CudaSystem& system)
{
    //Debug::DumpBatchedMemToFile("qFunc", qFuncGPU.gQFunction,
    //                            qFuncGPU.dataPerNode.Multiply(),
    //                            qFuncGPU.dataPerNode.Multiply() * qFuncGPU.nodeCount);

    distributions.UpdateDistributions(qFuncGPU.gQFunction, true,
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


    // Generate Distributions over this
    distributions = PWCDistStaticCPU2D(qFuncGPU.gQFunction,
                                       spatialCount,
                                       qFuncGPU.dataPerNode,
                                       true,
                                       system);

    qFuncGPU.gDistributions = distributions.DistributionGPU();
    return TracerError::OK;
}

void QFunctionCPU::DumpFunctionAsBinary(std::vector<Byte>& dataOut) const
{
    size_t size = qFuncGPU.dataPerNode.Multiply() * qFuncGPU.nodeCount * sizeof(float);
    dataOut.resize(size);

    CUDA_CHECK(cudaMemcpy(dataOut.data(), qFuncGPU.gQFunction,
                          size, cudaMemcpyDeviceToHost));
}