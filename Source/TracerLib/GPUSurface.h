#pragma once
/**

List of Sample Surface Definitions
and surface generation functions for sample primitives

*/

#include "RayLib/Ray.h"
#include "RayLib/Quaternion.h"

//struct BasicSurface
//{
//    private:        
//        QuatF   worldToTangent;     // World to tangent space transformation
//
//    protected:
//    public:
//        //// Constructors & Destructor
//        //__device__              GPUSurface(const QuatF& tbn);
//        //__device__              GPUSurface(const QuatF& tbn);
//        //                        GPUSurface(const GPUSurface&) = default;
//        //                        GPUSurface(GPUSurface&&) = default;
//        //GPUSurface&             operator=(const GPUSurface&) = default;
//        //GPUSurface&             operator=(GPUSurface&&) = default;
//        //                        ~GPUSurface() = default;
//
//        //__device__          
//        //// Functionality
//        //__device__ Vector3      NormalWorld() const;
//        //__device__ Vector3      TangentWorld() const;
//        //__device__ Vector3      BitangentWorld() const;
//        //
//        //__device__ Vector3      Normal() const;
//        //__device__ Vector3      Tangent() const;
//        //__device__ Vector3      Bitangent() const;
//
//        //__device__ RayF         ToTangent(const RayF&) const;
//        //__device__ RayF         FromTangent(const RayF&) const;
//
//        //__device__ float        DotN(const Vector3&) const;
//        //__device__ float        DotT(const Vector3&) const;
//        //__device__ float        DotB(const Vector3&) const;
//};

//
//
//__device__
//inline GPUSurface::GPUSurface(const QuatF& tbn)
//    : worldToTangent(tbn)
//{}
//
//__device__     
//inline GPUSurface::GPUSurface(const QuatF& tbn)
//    : worldToTangent(tbn)
//{}
//
//__device__ 
//inline Vector3 GPUSurface::NormalWorld() const
//{
//    Vector3 v = ZAxis;
//    return worldToTangent.Conjugate().ApplyRotation(v);
//}
//
//__device__
//inline Vector3 GPUSurface::TangentWorld() const
//{
//    Vector3 v = XAxis;
//    return worldToTangent.Conjugate().ApplyRotation(v);
//}
//
//__device__
//inline Vector3 GPUSurface::BitangentWorld() const
//{
//    Vector3 v = YAxis;
//    return worldToTangent.Conjugate().ApplyRotation(v);
//}
//
//__device__
//inline Vector3 GPUSurface::Normal() const
//{
//    return ZAxis;
//}
//
//__device__
//inline Vector3 GPUSurface::Tangent() const
//{
//    return XAxis;
//}
//
//__device__
//inline Vector3 GPUSurface::Bitangent() const
//{
//    return YAxis;
//}
//
//__device__
//inline RayF GPUSurface::ToTangent(const RayF& r) const
//{
//    return r.Transform(worldToTangent);
//}
//
//__device__
//inline RayF GPUSurface::FromTangent(const RayF& r) const
//{
//    return r.Transform(worldToTangent.Conjugate());
//}
//
//__device__ 
//inline float GPUSurface::DotN(const Vector3& v) const
//{
//    return v[2];
//}
//
//__device__
//inline float GPUSurface::DotT(const Vector3& v) const
//{
//    return v[0];
//}
//
//__device__
//inline float GPUSurface::DotB(const Vector3& v) const
//{
//    return v[1];
//}


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
    // World to tangent space transformation
    QuatF   worldToTangent;
};

struct UVSurface
{
    // World to tangent space transformation
    QuatF   worldToTangent;
    Vector2 uv;
};

//// Surface Functions
//template <class Primitive>
//__device__ __host__
//inline EmptySurface EmptySurfaceFromAny(const typename Primitive::PrimitiveData& pData,
//                                        const typename Primitive::HitData& hData,
//                                        PrimitiveId id)
//{
//    return EmptySurface{};
//}
//
//__device__ __host__
//inline BarySurface BarySurfaceFromTri(const GPUPrimitiveTriangle::PrimitiveData& pData,
//                                      const GPUPrimitiveTriangle::HitData& hData,
//                                      PrimitiveId id)
//{
//    Vector3 baryCoord = Vector3(hData[0],
//                                hData[1],
//                                1 - hData[0] - hData[1]);
//    return BarySurface{baryCoord};
//}
//
//__device__ __host__
//inline BasicSurface BasicSurfaceFromTri(const GPUPrimitiveTriangle::PrimitiveData& pData,
//                                        const GPUPrimitiveTriangle::HitData& hData,
//                                        PrimitiveId id)
//{
//
//    uint64_t index0 = pData.indexList[id * 3 + 0];
//    uint64_t index1 = pData.indexList[id * 3 + 1];
//    uint64_t index2 = pData.indexList[id * 3 + 2];
//
//    Vector3 n0 = pData.normalsV[index0];
//    Vector3 n1 = pData.normalsV[index1];
//    Vector3 n2 = pData.normalsV[index2];
//
//    Vector3 baryCoord = Vector3(hData[0],
//                                hData[1],
//                                1 - hData[0] - hData[1]);
//
//    Vector3 nAvg = (baryCoord[0] * n0 +
//                    baryCoord[1] * n1 +
//                    baryCoord[2] * n2);
//    return BasicSurface{nAvg};
//}
//
//__device__ __host__
//inline UVSurface UVSurfaceFromTri(const GPUPrimitiveTriangle::PrimitiveData& pData,
//                                  const GPUPrimitiveTriangle::HitData& hData,
//                                  PrimitiveId id)
//{
//
//    uint64_t index0 = pData.indexList[id * 3 + 0];
//    uint64_t index1 = pData.indexList[id * 3 + 1];
//    uint64_t index2 = pData.indexList[id * 3 + 2];
//
//    Vector4 n0 = pData.normalsV[index0];
//    Vector4 n1 = pData.normalsV[index1];
//    Vector4 n2 = pData.normalsV[index2];
//
//    Vector2 uv0 = Vector2(pData.positionsU[index0][3], n0[3]);
//    Vector2 uv1 = Vector2(pData.positionsU[index1][3], n1[3]);
//    Vector2 uv2 = Vector2(pData.positionsU[index2][3], n2[3]);
//
//    Vector3 baryCoord = Vector3(hData[0],
//                                hData[1],
//                                1 - hData[0] - hData[1]);
//
//    Vector3 nAvg = (baryCoord[0] * Vector3(n0) +
//                    baryCoord[1] * Vector3(n1) +
//                    baryCoord[2] * Vector3(n2));
//    Vector2 uvAvg = (baryCoord[0] * uv0 +
//                     baryCoord[1] * uv1 +
//                     baryCoord[2] * uv2);
//    return UVSurface{nAvg, uvAvg};
//}
//
//__device__ __host__
//inline BasicSurface BasicSurfaceFromSphr(const GPUPrimitiveSphere::PrimitiveData& pData,
//                                         const GPUPrimitiveSphere::HitData& hData,
//                                         PrimitiveId id)
//{
//    Vector4f data = pData.centerRadius[id + 0];
//    Vector3f center = data;
//    float r = data[3];
//
//    // Convert spherical hit to cartesian
//    Vector3 normal = Vector3(sin(hData[0]) * cos(hData[1]),
//                              sin(hData[0]) * sin(hData[1]),
//                              cos(hData[0]));   
//    return BasicSurface{normal};
//}
//
//__device__ __host__
//inline UVSurface UVSurfaceFromSphr(const GPUPrimitiveSphere::PrimitiveData& pData,
//                                   const GPUPrimitiveSphere::HitData& hData,
//                                   PrimitiveId id)
//{
//    Vector4f data = pData.centerRadius[id + 0];
//    Vector3f center = data;
//    float r = data[3];
//
//    // Convert spherical hit to cartesian
//    Vector3 normal = Vector3(sin(hData[0]) * cos(hData[1]),
//                             sin(hData[0]) * sin(hData[1]),
//                             cos(hData[0]));
//    // Gen UV    
//    Vector2 uv = hData;
//    // tetha is [-pi, pi], normalize
//    uv[0] = (uv[0] + MathConstants::Pi) * 0.5f * MathConstants::InvPi; 
//    // phi is [0, pi], normalize 
//    uv[1] /= MathConstants::Pi; 
//
//    return UVSurface{normal, uv};
//}
//
//__device__ __host__
//inline SphrSurface SphrSurfaceFromSphr(const GPUPrimitiveSphere::PrimitiveData& pData,
//                                       const GPUPrimitiveSphere::HitData& hData,
//                                       PrimitiveId id)
//{
//    return SphrSurface{hData};
//}