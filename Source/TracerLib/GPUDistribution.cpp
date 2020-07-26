#include "GPUDistribution.h"

__host__ 
Distribution1D::Distribution1D(const std::vector<float>& values)
{

}

__host__
const GPUDistribution2D& Distribution2D::DistributionGPU() const
{
    return gpuDistribution;
}


__host__ 
Distribution2D::Distribution2D(const std::vector<float>& values,
                               uint32_t width, uint32_t height)
{

}

__host__
const GPUDistribution1D& Distribution1D::DistributionGPU() const
{
    return gpuDistribution;
}

