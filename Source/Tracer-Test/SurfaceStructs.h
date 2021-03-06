#pragma once

#include "TracerLib/GPUPrimitiveSphere.h"
#include "TracerLib/GPUPrimitiveTriangle.h"
#include "TracerLib/GPUPrimitiveEmpty.h"

struct BarySurface
{
    Vector3 baryCoords;
};

struct SphrSurface
{
    Vector2 sphrCoords;
};

struct BasicSurface
{
    Vector3 normal;
};

struct EmptySurface
{};

// Surface Functions
__device__ __host__
inline BarySurface BarySurfaceFromTri(const GPUPrimitiveTriangle::PrimitiveData& pData,
                                      const GPUPrimitiveTriangle::HitData& hData,
                                      PrimitiveId id)
{
    Vector3 baryCoord = Vector3(hData[0],
                                hData[1],
                                1 - hData[0] - hData[1]);
    return {baryCoord};
}

__device__ __host__
inline BasicSurface BasicSurfaceFromTri(const GPUPrimitiveTriangle::PrimitiveData& pData,
                                        const GPUPrimitiveTriangle::HitData& hData,
                                        PrimitiveId id)
{

    uint64_t index0 = pData.indexList[id * 3 + 0];
    uint64_t index1 = pData.indexList[id * 3 + 1];
    uint64_t index2 = pData.indexList[id * 3 + 2];

    Vector3 n0 = pData.normalsV[index0];
    Vector3 n1 = pData.normalsV[index1];
    Vector3 n2 = pData.normalsV[index2];

    Vector3 baryCoord = Vector3(hData[0],
                                hData[1],
                                1 - hData[0] - hData[1]);

    Vector3 nAvg = (baryCoord[0] * n0 +
                    baryCoord[1] * n1 +
                    baryCoord[2] * n2);
    return {nAvg};
}

__device__ __host__
inline EmptySurface EmptySurfaceFromTri(const GPUPrimitiveTriangle::PrimitiveData& pData,
                                        const GPUPrimitiveTriangle::HitData& hData,
                                        PrimitiveId id)
{
    return {};
}

__device__ __host__
inline BasicSurface BasicSurfaceFromSphr(const GPUPrimitiveSphere::PrimitiveData& pData,
                                         const GPUPrimitiveSphere::HitData& hData,
                                         PrimitiveId id)
{
    Vector4f data = pData.centerRadius[id + 0];
    Vector3f center = data;
    float r = data[3];

    // Convert spherical hit to cartesian
    Vector3 normal = Vector3(sin(hData[0]) * cos(hData[1]),
                              sin(hData[0]) * sin(hData[1]),
                              cos(hData[0]));   
    return {normal};
}

__device__ __host__
inline EmptySurface EmptySurfaceFromSphr(const GPUPrimitiveSphere::PrimitiveData& pData,
                                         const GPUPrimitiveSphere::HitData& hData,
                                         PrimitiveId id)
{
    return {};
}

__device__ __host__
inline SphrSurface SphrSurfaceFromSphr(const GPUPrimitiveSphere::PrimitiveData& pData,
                                       const GPUPrimitiveSphere::HitData& hData,
                                       PrimitiveId id)
{
    return { hData };
}

__device__ __host__
inline EmptySurface EmptySurfaceFromEmpty(const GPUPrimitiveEmpty::PrimitiveData& pData,
                                          const GPUPrimitiveEmpty::HitData& hData,
                                          PrimitiveId id)
{
    return {};
}