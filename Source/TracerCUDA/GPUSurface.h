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

// Each Surface should have a member function of WorldNormal()
// For works to access it
// SFINAE
template <typename C, typename = void>
struct HasGetNormFunc : public std::false_type {};

template <typename C>
struct HasGetNormFunc <C,
    typename std::enable_if<std::is_same_v<decltype(&C::WorldNormal),
                                           Vector3f(C::*)() const>>::type
> : public std::true_type {};


struct EmptySurface
{
    __device__
    Vector3f WorldNormal() const { return Zero3f; };
};

struct BarySurface
{
    Vector3 baryCoords;

    __device__
    Vector3f WorldNormal() const { return Zero3f; };
};

struct SphrSurface
{
    Vector2 sphrCoords;

    __device__
    Vector3f WorldNormal() const { return Zero3f; };
};

struct BasicSurface
{
    // World to tangent space transformation
    QuatF worldToTangent;

    __device__
    Vector3f WorldNormal() const;
};

struct UVSurface
{
    // World to tangent space transformation
    QuatF   worldToTangent;
    Vector2 uv;

    __device__
    Vector3f WorldNormal() const;
};

static_assert(HasGetNormFunc<EmptySurface>::value,
              "EmptySurface do not have WorldNormal Function");
static_assert(HasGetNormFunc<BarySurface>::value,
              "BarySurface do not have WorldNormal Function");
static_assert(HasGetNormFunc<SphrSurface>::value,
              "SphrSurface do not have WorldNormal Function");
static_assert(HasGetNormFunc<BasicSurface>::value,
              "BasicSurface do not have WorldNormal Function");
static_assert(HasGetNormFunc<UVSurface>::value,
              "UVSurface do not have WorldNormal Function");

__device__ __forceinline__
Vector3f BasicSurface::WorldNormal() const
{
    return GPUSurface::NormalWorld(worldToTangent);
}

__device__ __forceinline__
Vector3f UVSurface::WorldNormal() const
{
    return GPUSurface::NormalWorld(worldToTangent);
}

// Meta Functions
// (Primitive invariant functions)
template <class HitData, class PrimData>
__device__
EmptySurface DefaultGenEmptySurface(const HitData&,
                                    const GPUTransformI&,
                                    PrimitiveId,
                                    const PrimData&)
{
    return EmptySurface{};
}

template <class HitData, class PrimData>
__device__
BasicSurface DefaultGenBasicSurface(const HitData&,
                                    const GPUTransformI& t,
                                    PrimitiveId,
                                    const PrimData&)
{
    return BasicSurface{t.ToLocalRotation()};
}

template <class HitData, class PrimData>
__device__
UVSurface DefaultGenUvSurface(const HitData&,
                              const GPUTransformI& t,
                              PrimitiveId,
                              const PrimData&)
{
    return UVSurface{t.ToLocalRotation(), Zero2f};
}