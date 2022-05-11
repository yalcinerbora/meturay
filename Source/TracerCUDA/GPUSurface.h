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
    inline Vector3 NormalToSpace(const QuatF& spaceToTangentTransform)
    {
        Vector3 v = ZAxis;
        return spaceToTangentTransform.Conjugate().ApplyRotation(v);
    }

    __device__
    inline Vector3 TangentToSpace(const QuatF& spaceToTangentTransform)
    {
        Vector3 v = YAxis;
        return spaceToTangentTransform.Conjugate().ApplyRotation(v);
    }

    __device__
    inline Vector3 BitangentToSpace(const QuatF& spaceToTangentTransform)
    {
        Vector3 v = XAxis;
        return spaceToTangentTransform.Conjugate().ApplyRotation(v);
    }

    __device__
    inline Vector3 ToTangent(const Vector3f& v,
                             const QuatF& toTangentTransform)
    {
        return toTangentTransform.ApplyRotation(v);
    }

    __device__
    inline Vector3 ToSpace(const Vector3f& v,
                           const QuatF& spaceToTangentTransform)
    {
        return spaceToTangentTransform.Conjugate().ApplyRotation(v);
    }

    __device__
    inline RayF ToSpace(const RayF& r,
                        const QuatF& spaceToTangentTransform)
    {
        return r.Transform(spaceToTangentTransform);
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
// TODO:
// I've added "worldPosition" "worldGeoNormal" functions
// but don't want to clutter using SFINAE to mandate the availability
// of the functions. Maybe after some time use concepts here
// when NVCC supports it (maybe it already supports? 2022)
struct EmptySurface
{
    __device__
    Vector3f WorldNormal() const { return Zero3f; };
    __device__
    Vector3f WorldPosition() const { return Zero3f; };
    __device__
    Vector3f WorldGeoNormal() const { return Zero3f; };
};

struct BarySurface
{
    Vector3 baryCoords;

    __device__
    Vector3f WorldNormal() const { return Zero3f; };
    __device__
    Vector3f WorldPosition() const { return Zero3f; };
    __device__
    Vector3f WorldGeoNormal() const { return Zero3f; };
};

struct SphrSurface
{
    Vector2 sphrCoords;

    __device__
    Vector3f WorldNormal() const { return Zero3f; };
    __device__
    Vector3f WorldPosition() const { return Zero3f; };
    __device__
    Vector3f WorldGeoNormal() const { return Zero3f; };
};

struct BasicSurface
{
    Vector3f    worldPosition;      // World surface location
    QuatF       worldToTangent;     // World to tangent space transformation
    Vector3f    worldGeoNormal;     // Geometric normal (useful when nudge the ray)
    bool        backSide;           // Returning the side of the surface (used on transmissive materials)
    // If a mesh does try to approximate a curved surface
    // (by smoothed normals), sometimes ray geometrically hit the surface
    // but in theory they shouldn't (actually this mesh is badly modeled but
    // and rays should self-intersect with the mesh but w/e)
    // this value approximates the curvature as a perfect sphere and creates
    // and offset between the chord(mesh surface) and that sphere
    // This is used to offset rays even more from the mesh
    float       curvatureOffset;


    __device__
    Vector3f WorldNormal() const;
    __device__
    Vector3f WorldPosition() const;
    __device__
    Vector3f WorldGeoNormal() const;
};

struct UVSurface
{

    Vector3f    worldPosition;  // World surface location
    QuatF       worldToTangent; // World to tangent space transformation
    Vector2     uv;             // Texture coordinates
    Vector3f    worldGeoNormal; // Geometric normal (useful when nudge the ray)
    bool        backSide;       // Returning the side of the surface (used on transmissive materials)
        // If a mesh does try to approximate a curved surface
    // (by smoothed normals), sometimes ray geometrically hit the surface
    // but in theory they shouldn't (actually this mesh is badly modeled but
    // and rays should self-intersect with the mesh but w/e)
    // this value approximates the curvature as a perfect sphere and creates
    // and offset between the chord(mesh surface) and that sphere
    // This is used to offset rays even more from the mesh
    float       curvatureOffset;

    __device__
    Vector3f WorldNormal() const;
    __device__
    Vector3f WorldPosition() const;
    __device__
    Vector3f WorldGeoNormal() const;
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

__device__ inline
Vector3f BasicSurface::WorldNormal() const
{
    return GPUSurface::NormalToSpace(worldToTangent);
}

__device__ inline
Vector3f BasicSurface::WorldPosition() const
{
    return worldPosition;
}

__device__ inline
Vector3f BasicSurface::WorldGeoNormal() const
{
    return worldGeoNormal;
}

__device__ inline
Vector3f UVSurface::WorldNormal() const
{
    return GPUSurface::NormalToSpace(worldToTangent);
}

__device__ inline
Vector3f UVSurface::WorldPosition() const
{
    return worldPosition;
}

__device__ inline
Vector3f UVSurface::WorldGeoNormal() const
{
    return worldGeoNormal;
}

// Meta Functions
// (Primitive invariant functions)
template <class HitData, class PrimData>
__device__
EmptySurface DefaultGenEmptySurface(const HitData&,
                                    const GPUTransformI&,
                                    const Vector3f&,
                                    PrimitiveId,
                                    const PrimData&)
{
    return EmptySurface{};
}

template <class HitData, class PrimData>
__device__
BasicSurface DefaultGenBasicSurface(const HitData&,
                                    const GPUTransformI& t,
                                    const Vector3f&,
                                    PrimitiveId,
                                    const PrimData&)
{
    return BasicSurface{Zero3f, t.ToLocalRotation(), Zero3f, false, 0.0f};
}

template <class HitData, class PrimData>
__device__
UVSurface DefaultGenUvSurface(const HitData&,
                              const GPUTransformI& t,
                              const Vector3f&,
                              PrimitiveId,
                              const PrimData&)
{
    return UVSurface{Zero3f, t.ToLocalRotation(), Zero2f, Zero3f, false, 0.0f};
}