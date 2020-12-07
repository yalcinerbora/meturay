#pragma once

#include <vector>
#include <cstdint>
#include <cuda.h>
#include <algorithm>
#include "RayLib/Vector.h"

#include "DeviceMemory.h"

class GPUDistribution1D
{
    friend class Distribution1D;

    private:
        const float*    dList   = nullptr;
        float*          dCDF    = nullptr;
        uint32_t        count   = 0;

    protected:
    public:
        // Construtors & Destructor
                                GPUDistribution1D() = default;
        __host__                GPUDistribution1D(float* dCDFList,
                                                  const float* dList,
                                                  uint32_t count);
                                GPUDistribution1D(const GPUDistribution1D&) = delete;
                                GPUDistribution1D(GPUDistribution1D&&) = default;
        GPUDistribution1D&      operator=(const GPUDistribution1D&) = delete;
        GPUDistribution1D&      operator=(GPUDistribution1D&&) = default;
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
        uint32_t            width = 0;
        uint32_t            height = 0;

    protected:
    public:
        // Construtors & Destructor
                            GPUDistribution2D() = default;
        __host__            GPUDistribution2D(float* dCDFList,
                                              const float* dList,
                                              uint32_t countX,
                                              uint32_t countY);
                            GPUDistribution2D(const GPUDistribution2D&) = delete;
                            GPUDistribution2D(GPUDistribution2D&&) = default;
        GPUDistribution2D&  operator=(const GPUDistribution2D&) = delete;
        GPUDistribution2D&  operator=(GPUDistribution2D&&) = default;
                            ~GPUDistribution2D() = default;

        // Interface                                
        __host__ __device__ 
        float       Sample(float& pdf, Vector2ui& index, 
                            const Vector2f& xi) const;
        __host__ __device__ 
        uint32_t    Width() const { return width; }
        uint32_t    Height() const { return height; }
};

class Distribution1D
{
    private:
        DeviceMemory                memory;
        GPUDistribution1D           gpuDistribution1D;
        GPUDistribution2D           gpuDistribution2D;

    protected:
    public:
        // Constructors & Destructor
                                    Distribution1D() = default;
                                    Distribution1D(const std::vector<float>& values);
                                    Distribution1D(const Distribution1D&) = delete;
                                    Distribution1D(Distribution1D&&) = default;
        Distribution1D&             operator=(const Distribution1D&) = delete;
        Distribution1D&             operator=(Distribution1D&&) = default;        
                                    ~Distribution1D() = default;

        const GPUDistribution1D&    DistributionGPU() const;

        // One dimensional distribution wrapper on 2D
        // in order to skip polymorphism on GPU
        const GPUDistribution2D&    DistributionGPU2D() const;
};

class Distribution2D
{
    private:
        DeviceMemory                memory;
        GPUDistribution2D           gpuDistribution;

    protected:
    public:
        // Constructors & Destructor
                                    Distribution2D() = default;
                                    Distribution2D(const std::vector<float>& values,
                                                   uint32_t width, uint32_t height);
                                    Distribution2D(const Distribution2D&) = delete;
                                    Distribution2D(Distribution2D&&) = default;
        Distribution2D&             operator=(const Distribution2D&) = delete;
        Distribution2D&             operator=(Distribution2D&&) = default;
                                    ~Distribution2D() = default;

        __host__
        const GPUDistribution2D&    DistributionGPU() const;
};

__host__
inline GPUDistribution1D::GPUDistribution1D(float* dCDFList,
                                            const float* dList,
                                            uint32_t count)
    : dCDF(dCDFList)
    , dList(dList)
    , count(count)
{
    //static constexpr uint32_t PARRALLEL_ALGO_THREHOLD = 1204;

    // TODO: Implement
    // Generate CDF
    // ....
}

__host__
inline float GPUDistribution1D::Sample(float& pdf, int& index, float xi) const
{
    // TODO: Implement
    pdf = 1.0f;
    index = static_cast<uint32_t>(xi * count);
    return 0.0f;
}

__host__
inline GPUDistribution2D::GPUDistribution2D(float* dCDFList,
                                            const float* dList,
                                            uint32_t countX,
                                            uint32_t countY)
    : width(countX)
    , height(countY)
{
    // TODO: Implement
    // Generate CDF
    // ....
}

__device__
inline float GPUDistribution2D::Sample(float& pdf,
                                       Vector2ui& index,
                                       const Vector2f& xi) const
{
   
    // TODO: Implement
    pdf = 1.0f;
    index = Vector2ui(static_cast<uint32_t>(width * xi[0]),
                      static_cast<uint32_t>(height * xi[1]));
    return 0.0f;
}