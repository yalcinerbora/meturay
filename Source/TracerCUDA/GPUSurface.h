#pragma once
/**

List of Sample Surface Definitions
and surface generation functions for sample primitives

*/

#include "RayLib/Ray.h"
#include "RayLib/Quaternion.h"

#include "GPUTransformI.h"

namespace GPUSurface
{
    __device__
    inline Vector3 NormalWorld(const QuatF& toTangentTransform)
    {
        Vector3 v = ZAxis;
        return toTangentTransform.Conjugate().ApplyRotation(v);
    }

    __device__
    inline Vector3 TangentWorld(const QuatF& toTangentTransform)
    {
        Vector3 v = YAxis;
        return toTangentTransform.Conjugate().ApplyRotation(v);
    }

    __device__
    inline Vector3 BitangentWorld(const QuatF& toTangentTransform)
    {
        Vector3 v = XAxis;
        return toTangentTransform.Conjugate().ApplyRotation(v);
    }

    __device__
    inline Vector3 ToTangent(const Vector3f& v,
                             const QuatF& toTangentTransform)
    {
        return toTangentTransform.ApplyRotation(v);
    }

    __device__
    inline Vector3 ToWorld(const Vector3f& v,
                           const QuatF& toTangentTransform)
    {
        return toTangentTransform.Conjugate().ApplyRotation(v);
    }

    __device__
    inline RayF ToTangent(const RayF& r,
                          const QuatF& toTangentTransform)
    {
        return r.Transform(toTangentTransform);
    }

    __device__
    inline RayF FromTangent(const RayF& r,
                            const QuatF& toTangentTransform)
    {
        return r.Transform(toTangentTransform.Conjugate());
    }

    __device__
    inline Vector3 Normal()
    {
        return ZAxis;
    }

    __device__
    inline Vector3 Tangent()
    {
        return XAxis;
    }

    __device__
    inline Vector3 Bitangent()
    {
        return YAxis;
    }

    __device__
    inline float DotN(const Vector3& v)
    {
        return v[2];
    }

    __device__
    inline float DotT(const Vector3& v)
    {
        return v[0];
    }

    __device__
    inline float DotB(const Vector3& v)
    {
        return v[1];
    }
}

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
    QuatF worldToTangent;
};

struct UVSurface
{
    // World to tangent space transformation
    QuatF   worldToTangent;
    Vector2 uv;
};


// Meta Functions
// (Primitive invariant functions)
template <class HitData, class PrimData>
__device__
EmptySurface GenEmptySurface(const HitData&,
                             const GPUTransformI&,
                             PrimitiveId,
                             const PrimData&)
{
    return EmptySurface{};
}

template <class HitData, class PrimData>
__device__
BasicSurface GenBasicSurface(const HitData&,
                             const GPUTransformI& t,
                             PrimitiveId,
                             const PrimData&)
{    
    return BasicSurface{t.ToLocalRotation()};
}