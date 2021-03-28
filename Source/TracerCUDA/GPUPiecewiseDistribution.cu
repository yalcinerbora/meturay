#include "GPUPiecewiseDistribution.cuh"
#include <numeric>

__host__ 
CPUDistGroupPiecewise1D::CPUDistGroupPiecewise1D(const std::vector<std::vector<float>>& pdfValues,
                                                 const CudaSystem& cudaSystem)
{
    // Gen Sizes
    std::vector<size_t> sizes(pdfValues.size());
    std::transform(pdfValues.cbegin(), pdfValues.cend(), sizes.begin(),
                   [](const std::vector<float>& vec)
                   {
                       return vec.size();
                   });
    std::reduce(sizes.begin(), sizes.end)
    // Allocate Memory
    size_t 

}

__host__
const GPUDistPiecewise1D& CPUDistGroupPiecewise1D::DistributionGPU(uint32_t index) const
{
    return gpuDistributions[index];
}

__host__ 
CPUDistGroupPiecewise2D::CPUDistGroupPiecewise2D(const std::vector<std::vector<float>>& pdfValues,
                                                 const std::vector<Vector2ui> dimensions)
{

}

__host__
const GPUDistPiecewise2D& CPUDistGroupPiecewise2D::DistributionGPU(uint32_t index) const
{
    return gpuDistributions[index];
}

