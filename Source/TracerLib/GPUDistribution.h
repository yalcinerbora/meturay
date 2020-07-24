#pragma once

#include <vector>
#include <cstdint>
#include <cuda.h>
#include "RayLib/Vector.h"

#include "DeviceMemory.h"

// Some Kernels


class GPUDistribution1D
{
    friend class Distribution1D;

    private:
        const float*    dList;
        float*          dCDF;
        uint32_t        count;

    protected:
    public:
        // Construtors & Destructor
        __host__        GPUDistribution1D(/*float* dCDFList,
                                          const float* dList, 
                                          uint32_t count*/);
                        ~GPUDistribution1D() = default;

        __host__ __device__ 
        float           Sample(float& pdf, int& index, float xi) const;        
        __host__ __device__ 
        uint32_t        Count() const { return count; }
};

class GPUDistribution2D
{
    friend class Distribution2D;
    private:
        GPUDistribution1D   dDistributionsY;
        GPUDistribution1D*  dDistributionsX;
        uint32_t            count;

    protected:
    public:
        // Construtors & Destructor
        __host__        GPUDistribution2D(/*float* dCDFList,
                                      const float* dList,
                                      uint32_t count*/);
                        ~GPUDistribution2D() = default;

        // Interface                                
        __host__ __device__ 
        float       Sample(float& pdf, Vector2ui& index, 
                            const Vector2f& xi) const;
        __host__ __device__ 
        uint32_t    Count() const { return count; }
};

class Distribution1D
{
    private:
        DeviceMemory                memory;
        GPUDistribution1D           gpuDistribution;

    protected:
    public:
        __host__                    Distribution1D(const std::vector<float>& values);
                                    ~Distribution1D() = default;
        __host__
        const GPUDistribution1D&    DistributionGPU() const;
};

class Distribution2D
{
    private:
        DeviceMemory                memory;
        GPUDistribution2D           gpuDistribution;

    protected:
    public:
        __host__                    Distribution2D(const std::vector<float>& values,
                                                   uint32_t width,
                                                   uint32_t height);
                                    ~Distribution2D() = default;

        __host__
        const GPUDistribution2D&    DistributionGPU() const;
};

__device__ GPUDistribution1D::GPUDistribution1D(/*float* dCDFList,
                                                const float* dList,
                                                uint32_t count*/)
{

}

__device__
float GPUDistribution1D::Sample(float& pdf, int& index, float xi) const
{
    return 0.0f;
}

__device__ GPUDistribution2D::GPUDistribution2D(/*float* dCDFList,
                                                const float* dList,
                                                uint32_t countX,
                                                uint32_t countY*/)
{

}

__device__
float GPUDistribution2D::Sample(float& pdf,
                                Vector2ui& index,
                                const Vector2f& xi) const
{
    return 0.0f;
}