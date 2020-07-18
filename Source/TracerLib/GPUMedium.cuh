#pragma once

#include <cuda.h>

#include "RayLib/Vector.h"

// Default medium is a vacuum with IoR=0
static constexpr uint32_t DEFAULT_MEDIUM_INDEX = 0;

// GPUMedium represents homogenous medium
class GPUMedium
{
    private:
        Vector3         sigmaA;
        Vector3         sigmaS;
        Vector3         sigmaT;
        float           ior;
        float           phase;

        uint32_t        id;

    public:
        // Constructors & Destructor
        __host__ __device__         GPUMedium();
        __host__ __device__         GPUMedium(float sA,
                                              float sS,
                                              float ior,
                                              float p,
                                              uint32_t id);
                                    ~GPUMedium() = default;

        // Funcs
       __device__ Vector3           SigmaA() const;
       __device__ Vector3           SigmaS() const;
       __device__ Vector3           SigmaT() const;
       __device__ float             IOR() const;
       __device__ float             Phase() const;
       __device__ uint32_t          ID() const;

       __device__ Vector3           Transmittance(float distance) const;
};

__host__ __device__
inline GPUMedium::GPUMedium()
    : sigmaA(0.0f)
    , sigmaS(0.0f)
    , sigmaT(0.0f)
    , ior(1.0f)
    , phase(0)
    , id(DEFAULT_MEDIUM_INDEX)
{}

__host__ __device__
inline GPUMedium::GPUMedium(float sA,
                            float sS,
                            float ior,
                            float p,
                            uint32_t id)
    : sigmaA(sA)
    , sigmaS(sS)
    , sigmaT(sA + sS)
    , ior(ior)
    , phase(p)
    , id(id)
{}

__device__ inline Vector3 GPUMedium::SigmaA() const{return sigmaA;}
__device__ inline Vector3 GPUMedium::SigmaS() const { return sigmaS; }
__device__ inline Vector3 GPUMedium::SigmaT() const { return sigmaT; }
__device__ inline float GPUMedium::IOR() const { return ior; }
__device__ inline float GPUMedium::Phase() const { return phase; }
__device__ inline uint32_t GPUMedium::ID() const { return id; }

__device__
inline Vector3 GPUMedium::Transmittance(float distance) const
{
    Vector3 result = -sigmaT * distance;
    result[0] = exp(result[0]);
    result[1] = exp(result[1]);
    result[2] = exp(result[2]);
    return result;
}