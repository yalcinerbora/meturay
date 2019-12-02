#pragma once
/**

Default Sphere Implementation
One of the fundamental functional types.

Has two types of data
Position and radius.

All of them should be provided

*/

#include <map>

#include "DefaultLeaf.h"
#include "GPUPrimitiveP.cuh"
#include "DeviceMemory.h"
#include "TypeTraits.h"

#include "RayLib/PrimitiveDataTypes.h"
#include "RayLib/Vector.h"
#include "RayLib/Sphere.h"

// Sphere memory layout
struct SphereData
{
    const Vector4f* centerRadius;
};

// Hit of sphere is spherical coordinates
using SphereHit = Vector2f;

// Sphere Hit Acceptance
__device__ __host__
inline HitResult SphereClosestHit(// Output
                                  HitKey& newMat,
                                  PrimitiveId& newPrimitive,
                                  SphereHit& newHit,
                                  // I-O
                                  RayReg& rayData,
                                  // Input
                                  const DefaultLeaf& leaf,
                                  const SphereData& primData)
{
    // Get Packed data and unpack
    Vector4f data = primData.centerRadius[leaf.primitiveId];
    Vector3f center = data;
    float radius = data[3];

    // Do Intersecton test
    Vector3 pos; float newT;
    bool intersects = rayData.ray.IntersectsSphere(pos, newT, center, radius);

    // Check if the hit is closer
    bool closerHit = intersects && (newT < rayData.tMax);
    if(closerHit)
    {
        rayData.tMax = newT;
        newMat = leaf.matId;
        newPrimitive = leaf.primitiveId;

        // Gen Spherical Coords (R can be fetched using primitiveId)
        Vector3 relativeCoord = pos - center;
        float tetha = acos(relativeCoord[2] / radius);
        float phi = atan2(relativeCoord[1], relativeCoord[0]);
        newHit = Vector2(tetha, phi);
    }
    return HitResult{false, closerHit};
}

__device__ __host__
inline AABB3f GenerateAABBSphere(PrimitiveId primitiveId, const SphereData& primData)
{
    // Get Packed data and unpack
    Vector4f data = primData.centerRadius[primitiveId];
    Vector3f center = data;
    float radius = data[3];

    return Sphere::BoundingBox(center, radius);
}

__device__ __host__
inline float GenerateAreaSphere(PrimitiveId primitiveId, const SphereData& primData)
{
    Vector4f data = primData.centerRadius[primitiveId];
    float radius = data[3];

    // Surface area is related to radius only (wrt of its square)
    // TODO: check if this is a good estimation
    return radius * radius;
}

class GPUPrimitiveSphere final
    : public GPUPrimitiveGroup<SphereHit, SphereData, DefaultLeaf,
                               SphereClosestHit, GenerateDefaultLeaf,
                               GenerateAABBSphere, GenerateAreaSphere>
{
    public:
        static constexpr const char*            TypeName() { return "Sphere"; }

        static constexpr PrimitiveDataLayout    POS_LAYOUT = PrimitiveDataLayout::FLOAT_3;
        static constexpr PrimitiveDataLayout    RADUIS_LAYOUT = PrimitiveDataLayout::FLOAT_1;

    private:
        DeviceMemory                            memory;

        // List of ranges for each batch
        uint64_t                                totalPrimitiveCount;
        std::map<uint32_t, Vector2ul>           batchRanges;
        std::map<uint32_t, AABB3>               batchAABBs;

    public:
        // Constructors & Destructor
                                                GPUPrimitiveSphere();
                                                ~GPUPrimitiveSphere() = default;

        // Interface
        // Pirmitive type is used for delegating scene info to this class
        const char*                             Type() const override;
        // Allocates and Generates Data
        SceneError                              InitializeGroup(const NodeListing& surfaceDatalNodes, double time,
                                                                const SurfaceLoaderGeneratorI& loaderGen,
                                                                const std::string& scenePath) override;
        SceneError                              ChangeTime(const NodeListing& surfaceDatalNodes, double time,
                                                           const SurfaceLoaderGeneratorI& loaderGen,
                                                           const std::string& scenePath) override;

        // Access primitive range from Id
        Vector2ul                               PrimitiveBatchRange(uint32_t surfaceDataId) const override;
        AABB3                                   PrimitiveBatchAABB(uint32_t surfaceDataId) const override;

        // Error check
        // Queries in order to check if this primitive group supports certain primitive data
        // Material may need that data
        bool                                    CanGenerateData(const std::string& s) const override;
};

static_assert(IsTracerClass<GPUPrimitiveSphere>::value,
              "GPUPrimitiveSphere is not a Tracer Class.");