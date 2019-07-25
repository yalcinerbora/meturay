#pragma once
/**

Default Triangle Implementation

Has three types of data
Position, Normal and UV.

These data are indirected by a single index (like DirectX and OpenGL)

All of them should be provided

*/

#include <map>

#include "RayLib/Vector.h"
#include "RayLib/Triangle.h"

#include "DefaultLeaf.h"
#include "GPUPrimitiveP.cuh"
#include "DeviceMemory.h"
#include "TypeTraits.h"

class SurfaceDataLoaderI;
using SurfaceDataLoaders = std::vector<std::unique_ptr<SurfaceDataLoaderI>>;

// Triangle Memory Layout
struct TriData
{
    const Vector4f* positionsU;
    const Vector4f* normalsV;
    const uint32_t* indexList;
};

// Triangle Hit is barycentric coordinates
// c is (1-a-b) thus it is not stored.
using TriangleHit = Vector2f;

// Triangle Hit Acceptance
__device__ __host__
inline HitResult TriangleClosestHit(// Output
                                    HitKey& newMat,
                                    PrimitiveId& newPrimitive,
                                    TriangleHit& newHit,
                                    // I-O
                                    RayReg& rayData,
                                    // Input
                                    const DefaultLeaf& leaf,
                                    const TriData& primData)
{
    // Get Position
    uint32_t index0 = primData.indexList[leaf.primitiveId * 3 + 0];
    uint32_t index1 = primData.indexList[leaf.primitiveId * 3 + 1];
    uint32_t index2 = primData.indexList[leaf.primitiveId * 3 + 2];

    Vector3 position0 = primData.positionsU[index0];
    Vector3 position1 = primData.positionsU[index1];
    Vector3 position2 = primData.positionsU[index2];

    // Do Intersecton test
    Vector3 baryCoords; float newT;
    bool intersects = rayData.ray.IntersectsTriangle(baryCoords, newT,
                                                     position0,
                                                     position1,
                                                     position2,
                                                     false);

    // Check if the hit is closer
    bool closerHit = intersects && (newT < rayData.tMax);
    if(closerHit)
    {
        rayData.tMax = newT;
        newMat = leaf.matId;
        newPrimitive = leaf.primitiveId;
        newHit = Vector2(baryCoords[0], baryCoords[1]);
    }
    return HitResult{false, closerHit};
}


__device__ __host__
inline AABB3f GenerateAABBTriangle(PrimitiveId primitiveId, const TriData& primData)
{
    // Get Position
    uint32_t index0 = primData.indexList[primitiveId * 3 + 0];
    uint32_t index1 = primData.indexList[primitiveId * 3 + 1];
    uint32_t index2 = primData.indexList[primitiveId * 3 + 2];

    Vector3 position0 = primData.positionsU[index0];
    Vector3 position1 = primData.positionsU[index1];
    Vector3 position2 = primData.positionsU[index2];

    return Triangle::BoundingBox(position0, position1, position2);
}

__device__ __host__
inline float GenerateAreaTriangle(PrimitiveId primitiveId, const TriData& primData)
{
    // Get Position
    uint32_t index0 = primData.indexList[primitiveId * 3 + 0];
    uint32_t index1 = primData.indexList[primitiveId * 3 + 1];
    uint32_t index2 = primData.indexList[primitiveId * 3 + 2];

    Vector3 position0 = primData.positionsU[index0];
    Vector3 position1 = primData.positionsU[index1];
    Vector3 position2 = primData.positionsU[index2];

    // CCW
    Vector3 vec0 = position1 - position0;
    Vector3 vec1 = position2 - position0;

    return Cross(vec0, vec1).Length() * 0.5f;
}

class GPUPrimitiveTriangle final
    : public GPUPrimitiveGroup<TriangleHit, TriData, DefaultLeaf,
                               TriangleClosestHit, GenerateDefaultLeaf,
                               GenerateAABBTriangle, GenerateAreaTriangle>
{
    public:
        static constexpr const char*            TypeName() { return "Triangle"; }

    private:
        DeviceMemory                            memory;

        // List of ranges for each batch
        uint64_t                                totalPrimitiveCount;
        uint64_t                                totalDataCount;

        std::map<uint32_t, Vector2ul>           batchRanges;
        std::map<uint32_t, Vector2ul>           batchDataRanges;
        std::map<uint32_t, AABB3>               batchAABBs;

    protected:
    public:
        // Constructors & Destructor
                                                GPUPrimitiveTriangle();
                                                ~GPUPrimitiveTriangle() = default;

        // Interface
        // Pirmitive type is used for delegating scene info to this class
        const char*                             Type() const override;
        // Allocates and Generates Data
        SceneError                              InitializeGroup(const NodeListing& surfaceDataNodes, double time,
                                                                const SurfaceLoaderGeneratorI&) override;
        SceneError                              ChangeTime(const NodeListing& surfaceDataNodes, double time,
                                                           const SurfaceLoaderGeneratorI&) override;

        // Access primitive range from Id
        Vector2ul                               PrimitiveBatchRange(uint32_t surfaceDataId) const override;
        AABB3                                   PrimitiveBatchAABB(uint32_t surfaceDataId) const override;

        // Error check
        // Queries in order to check if this primitive group supports certain primitive data
        // Material may need that data
        bool                                        CanGenerateData(const std::string& s) const override;
};

static_assert(IsTracerClass<GPUPrimitiveTriangle>::value,
              "GPUPrimitiveTriangle is not a Tracer Class.");