#pragma once
/**

List of Sample Surface Definitions
and surface generation functions for sample primitives

*/

#include "GPUPrimitiveSphere.h"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveEmpty.h"

struct EmptySurface
{};

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

struct UVSurface
{
    Vector3 normal;
    Vector2 uv;
};

struct TangentUVSurface
{
    Vector4 normalU;
    Vector4 tangentV;
};

// Surface Functions
template <class Primitive>
__device__ __host__
inline EmptySurface EmptySurfaceFromAny(const typename Primitive::PrimitiveData& pData,
                                        const typename Primitive::HitData& hData,
                                        PrimitiveId id)
{
    return EmptySurface{};
}

__device__ __host__
inline BarySurface BarySurfaceFromTri(const GPUPrimitiveTriangle::PrimitiveData& pData,
                                      const GPUPrimitiveTriangle::HitData& hData,
                                      PrimitiveId id)
{
    Vector3 baryCoord = Vector3(hData[0],
                                hData[1],
                                1 - hData[0] - hData[1]);
    return BarySurface{baryCoord};
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
    return BasicSurface{nAvg};
}

__device__ __host__
inline UVSurface UVSurfaceFromTri(const GPUPrimitiveTriangle::PrimitiveData& pData,
                                  const GPUPrimitiveTriangle::HitData& hData,
                                  PrimitiveId id)
{

    uint64_t index0 = pData.indexList[id * 3 + 0];
    uint64_t index1 = pData.indexList[id * 3 + 1];
    uint64_t index2 = pData.indexList[id * 3 + 2];

    Vector4 n0 = pData.normalsV[index0];
    Vector4 n1 = pData.normalsV[index1];
    Vector4 n2 = pData.normalsV[index2];

    Vector2 uv0 = Vector2(pData.positionsU[index0][3], n0[3]);
    Vector2 uv1 = Vector2(pData.positionsU[index1][3], n1[3]);
    Vector2 uv2 = Vector2(pData.positionsU[index2][3], n2[3]);

    Vector3 baryCoord = Vector3(hData[0],
                                hData[1],
                                1 - hData[0] - hData[1]);

    Vector3 nAvg = (baryCoord[0] * Vector3(n0) +
                    baryCoord[1] * Vector3(n1) +
                    baryCoord[2] * Vector3(n2));
    Vector2 uvAvg = (baryCoord[0] * uv0 +
                     baryCoord[1] * uv1 +
                     baryCoord[2] * uv2);
    return UVSurface{nAvg, uvAvg};
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
    return BasicSurface{normal};
}

__device__ __host__
inline UVSurface UVSurfaceFromSphr(const GPUPrimitiveSphere::PrimitiveData& pData,
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
    // Gen UV    
    Vector2 uv = hData;
    // tetha is [-pi, pi], normalize
    uv[0] = (uv[0] + MathConstants::Pi) * 0.5f * MathConstants::InvPi; 
    // phi is [0, pi], normalize 
    uv[1] /= MathConstants::Pi; 

    return UVSurface{normal, uv};
}

__device__ __host__
inline SphrSurface SphrSurfaceFromSphr(const GPUPrimitiveSphere::PrimitiveData& pData,
                                       const GPUPrimitiveSphere::HitData& hData,
                                       PrimitiveId id)
{
    return SphrSurface{hData};
}