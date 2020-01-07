#include "GPUEventEstimatorBasic.h"

const char* GPUEventEstimatorBasic::Type() const
{
    return TypeName();
}

void GPUEventEstimatorBasic::Construct(const CudaSystem&)
{
    size_t size = lightInfo.size() * sizeof(EstimatorInfo);
   
    // Just memcpy CPU Estimator info to GPU
    memory = std::move(DeviceMemory(size));
    CUDA_CHECK(cudaMemcpy(memory, lightInfo.data(), size, cudaMemcpyHostToDevice));

    // Set Light Count
    dData.dLights = static_cast<const EstimatorInfo*>(memory);
    dData.lightCount = static_cast<uint32_t>(lightInfo.size());
}