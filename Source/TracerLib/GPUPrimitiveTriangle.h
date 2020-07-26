#pragma once
/**

Default Triangle Implementation

Has three types of data
Position, Normal and UV.

These data are indirected by a single index (like DirectX and OpenGL)

All of them should be provided

*/

#include <map>

#include "RayLib/PrimitiveDataTypes.h"
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
    const uint64_t* indexList;
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
    //if(leaf.matId.value == 0x2000002)
        //printf("PrimId %llu, MatId %x\n", leaf.primitiveId, leaf.matId.value);

    // Get Position
    uint64_t index0 = primData.indexList[leaf.primitiveId * 3 + 0];
    uint64_t index1 = primData.indexList[leaf.primitiveId * 3 + 1];
    uint64_t index2 = primData.indexList[leaf.primitiveId * 3 + 2];

    Vector3 position0 = primData.positionsU[index0];
    Vector3 position1 = primData.positionsU[index1];
    Vector3 position2 = primData.positionsU[index2];

    // Do Intersecton test
    Vector3 baryCoords; float newT;
    bool intersects = rayData.ray.IntersectsTriangle(baryCoords, newT,
                                                     position0,
                                                     position1,
                                                     position2,
                                                     true);
    // Check if the hit is closer
    bool closerHit = intersects && (newT < rayData.tMax);
    if(closerHit)
    {
        rayData.tMax = newT;
        newMat = leaf.matId;
        newPrimitive = leaf.primitiveId;
        newHit = Vector2(baryCoords[0], baryCoords[1]);
    }
    //printf("ray dir{%f, %f, %f} "
    //       "old %f new %f --- Testing Mat: %x -> {%s, %s}\n",
    //       rayData.ray.getDirection()[0],
    //       rayData.ray.getDirection()[1],
    //       rayData.ray.getDirection()[2],

    //       oldT, newT,
    //       leaf.matId.value, 
    //       closerHit ? "Close!" : "      ",
    //       intersects ? "Intersects!" : "           ");

    return HitResult{false, closerHit};
}


__device__ __host__
inline AABB3f GenerateAABBTriangle(PrimitiveId primitiveId, const TriData& primData)
{
    // Get Position
    uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
    uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
    uint64_t index2 = primData.indexList[primitiveId * 3 + 2];

    Vector3 position0 = primData.positionsU[index0];
    Vector3 position1 = primData.positionsU[index1];
    Vector3 position2 = primData.positionsU[index2];

    return Triangle::BoundingBox(position0, position1, position2);
}

__device__ __host__
inline float GenerateAreaTriangle(PrimitiveId primitiveId, const TriData& primData)
{
    // Get Position
    uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
    uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
    uint64_t index2 = primData.indexList[primitiveId * 3 + 2];

    Vector3 position0 = primData.positionsU[index0];
    Vector3 position1 = primData.positionsU[index1];
    Vector3 position2 = primData.positionsU[index2];

    // CCW
    Vector3 vec0 = position1 - position0;
    Vector3 vec1 = position2 - position0;

    return Cross(vec0, vec1).Length() * 0.5f;
}

__device__ __host__
inline Vector3 GenerateCenterTriangle(PrimitiveId primitiveId, const TriData& primData)
{
    // Get Position
    uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
    uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
    uint64_t index2 = primData.indexList[primitiveId * 3 + 2];

    Vector3 position0 = primData.positionsU[index0];
    Vector3 position1 = primData.positionsU[index1];
    Vector3 position2 = primData.positionsU[index2];

    return position0 * 0.33333f +
        position1 * 0.33333f +
        position2 * 0.33333f;
}

class GPUPrimitiveTriangle final
    : public GPUPrimitiveGroup<TriangleHit, TriData, DefaultLeaf,
                               TriangleClosestHit, GenerateDefaultLeaf,
                               GenerateAABBTriangle, GenerateAreaTriangle,
                               GenerateCenterTriangle>
{
    public:
        static constexpr const char*            TypeName() { return "Triangle"; }

        static constexpr PrimitiveDataLayout    POS_LAYOUT = PrimitiveDataLayout::FLOAT_3;
        static constexpr PrimitiveDataLayout    UV_LAYOUT = PrimitiveDataLayout::FLOAT_2;
        static constexpr PrimitiveDataLayout    NORMAL_LAYOUT = PrimitiveDataLayout::FLOAT_3;
        static constexpr PrimitiveDataLayout    INDEX_LAYOUT = PrimitiveDataLayout::UINT64_1;

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
                                                                const SurfaceLoaderGeneratorI&,
                                                                const std::string& scenePath) override;
        SceneError                              ChangeTime(const NodeListing& surfaceDataNodes, double time,
                                                           const SurfaceLoaderGeneratorI&,
                                                           const std::string& scenePath) override;
        // Provides data to Event Estimator
        bool                                    HasPrimitive(uint32_t surfaceDataId) const override;
        SceneError                              GenerateLights(std::vector<CPULight>&,
                                                               const GPUDistribution2D&,
                                                               HitKey key,
                                                               uint32_t surfaceDataId,
                                                               const Matrix4x4& transform) const override;
        // Access primitive range from Id
        Vector2ul                               PrimitiveBatchRange(uint32_t surfaceDataId) const override;
        AABB3                                   PrimitiveBatchAABB(uint32_t surfaceDataId) const override;

        // Error check
        // Queries in order to check if this primitive group supports certain primitive data
        // Material may need that data
        bool                                    CanGenerateData(const std::string& s) const override;
};

static_assert(IsTracerClass<GPUPrimitiveTriangle>::value,
              "GPUPrimitiveTriangle is not a Tracer Class.");