#pragma once
/**

Default Triangle Implementation

Has three types of data
Position, Normal and UV.

These data are indirected by a single index (like DirectX and OpenGL)

All of them should be provided

*/

#include <map>
#include <tuple>
#include <vector>

#include "RayLib/PrimitiveDataTypes.h"
#include "RayLib/Vector.h"
#include "RayLib/Triangle.h"

#include "GPUPrimitiveP.cuh"
#include "GPUTransform.h"
#include "GPUSurface.h"
#include "DefaultLeaf.h"
#include "DeviceMemory.h"
#include "TypeTraits.h"

class SurfaceDataLoaderI;
using SurfaceDataLoaders = std::vector<std::unique_ptr<SurfaceDataLoaderI>>;

// Triangle Memory Layout
struct TriData
{
    // Kinda Perf Hog but most memory efficient
    // Binary search cull face for each prim
    const bool* cullFace;
    const uint64_t* primOffsets;
    uint32_t primBatchCount;
    // TODO: add alpha map

    const Vector3f* positions;
    const QuatF*    tbnRotations;
    const Vector2*  uvs;
    const uint64_t* indexList;
};

// Triangle Hit is barycentric coordinates
// c is (1-a-b) thus it is not stored.
using TriangleHit = Vector2f;

struct TriFunctions
{

    // Triangle Hit Acceptance
    __device__ __host__
    static inline HitResult Hit(// Output
                                HitKey& newMat,
                                PrimitiveId& newPrim,
                                TriangleHit& newHit,
                                // I-O
                                RayReg& rayData,
                                // Input
                                const GPUTransformI& transform,
                                const DefaultLeaf& leaf,
                                const TriData& primData)
    {
        // Simple Binary Search to determine
        // cull flag from primitiveId
        auto BinSearchCull = [&primData](PrimitiveId id)
        {
            int32_t start = 0;
            int32_t end = primData.primBatchCount;
            while(start <= end)
            {
                int32_t mid = (start + end) / 2;            
                uint64_t current = primData.primOffsets[mid];
                uint64_t next = primData.primOffsets[mid + 1];
                if(id >= current && id < next)
                    return primData.cullFace[mid];
                else if(id < current)
                    end = mid - 1;
                else if(id >= next)
                    start = mid + 1;            
            }
            // Default to true
            return true;
        };

        //if(leaf.matId.value == 0x2000002)
            //printf("PrimId %llu, MatId %x\n", leaf.primitiveId, leaf.matId.value);

        // Get Position
        uint64_t index0 = primData.indexList[leaf.primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[leaf.primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[leaf.primitiveId * 3 + 2];

        Vector3 position0 = primData.positions[index0];
        Vector3 position1 = primData.positions[index1];
        Vector3 position2 = primData.positions[index2];

        bool cull = BinSearchCull(leaf.primitiveId);

        // Do Intersecton test on local space
        RayF r = transform.WorldToLocal(rayData.ray);
        Vector3 baryCoords; float newT;
        bool intersects = r.IntersectsTriangle(baryCoords, newT,
                                               position0,
                                               position1,
                                               position2,
                                               cull);
        // Check if the hit is closer
        bool closerHit = intersects && (newT < rayData.tMax);
        if(closerHit)
        {
            rayData.tMax = newT;
            newMat = leaf.matId;
            newPrim = leaf.primitiveId;
            newHit = {baryCoords[0], baryCoords[1]};
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
    static inline AABB3f AABB(const GPUTransformI& transform,
                              //
                              PrimitiveId primitiveId, 
                              const TriData& primData)
    {
        // Get Position
        uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[primitiveId * 3 + 2];

        Vector3 position0 = primData.positions[index0];
        Vector3 position1 = primData.positions[index1];
        Vector3 position2 = primData.positions[index2];

        position0 = transform.LocalToWorld(position0);
        position1 = transform.LocalToWorld(position1);
        position2 = transform.LocalToWorld(position2);

        return Triangle::BoundingBox(position0, position1, position2);
    }

    __device__ __host__
    static inline float Area(PrimitiveId primitiveId, const TriData& primData)
    {
        // Get Position
        uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[primitiveId * 3 + 2];

        Vector3 position0 = primData.positions[index0];
        Vector3 position1 = primData.positions[index1];
        Vector3 position2 = primData.positions[index2];

        // CCW
        Vector3 vec0 = position1 - position0;
        Vector3 vec1 = position2 - position0;

        return Cross(vec0, vec1).Length() * 0.5f;
    }

    __device__ __host__
    static inline Vector3 Center(PrimitiveId primitiveId, const TriData& primData)
    {
        // Get Position
        uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[primitiveId * 3 + 2];

        Vector3 position0 = primData.positions[index0];
        Vector3 position1 = primData.positions[index1];
        Vector3 position2 = primData.positions[index2];

        return (position0 * 0.33333f +
                position1 * 0.33333f +
                position2 * 0.33333f);
    }

    static constexpr auto Leaf = GenerateDefaultLeaf<TriData>;
};

class GPUPrimitiveTriangle;

struct TriangleSurfaceGenerator
{
    __device__ __host__
    static inline BasicSurface GenBasicSurface(const TriangleHit& baryCoords,
                                               const GPUTransformI& transform,
                                               //
                                               PrimitiveId primitiveId,
                                               const TriData& primData)
    {
        float c = 1 - baryCoords[0] - baryCoords[1];

        uint64_t i0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t i1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t i2 = primData.indexList[primitiveId * 3 + 2];

        QuatF q0 = primData.tbnRotations[i0];
        QuatF q1 = primData.tbnRotations[i1];
        QuatF q2 = primData.tbnRotations[i2];
        QuatF tbn = Quat::BarySLerp(q0, q1, q2,
                                    baryCoords[0],
                                    baryCoords[1]);
        tbn = tbn * transform.ToLocalRotation();        
        return BasicSurface{tbn};
    }

    __device__ __host__
    static inline BarySurface GenBarySurface(const TriangleHit& baryCoords,
                                             const GPUTransformI& transform,
                                             //
                                             PrimitiveId primitiveId,
                                             const TriData& primData)
    {
        float c = 1.0f - baryCoords[0] - baryCoords[1];
        return BarySurface{Vector3(baryCoords[0], baryCoords[1], c)};
    }

    __device__ __host__
    static inline UVSurface GenUVSurface(const TriangleHit& baryCoords,
                                         const GPUTransformI& transform,
                                         //
                                         PrimitiveId primitiveId,
                                         const TriData& primData)
    {
        BasicSurface bs = GenBasicSurface(baryCoords, transform,
                                          primitiveId, primData);

        float c = 1 - baryCoords[0] - baryCoords[1];

        uint64_t i0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t i1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t i2 = primData.indexList[primitiveId * 3 + 2];

        Vector2 uv0 = primData.uvs[i0];
        Vector2 uv1 = primData.uvs[i1];
        Vector2 uv2 = primData.uvs[i2];

        Vector2 uv = (uv0 * baryCoords[0] +
                      uv1 * baryCoords[1] + 
                      uv2 * c);

        return UVSurface{bs.worldToTangent, uv};
    }
   
    template <class Surface, SurfaceFunc<Surface, TriangleHit, TriData> SF>
    struct SurfaceFunctionType
    {
        using type = Surface;
        static constexpr auto SurfaceGeneratorFunction = SF;
    };

    static constexpr auto GeneratorFunctionList = 
        std::make_tuple(SurfaceFunctionType<EmptySurface, GenEmptySurface<TriangleHit, TriData>>{},
                        SurfaceFunctionType<BasicSurface, GenBasicSurface>{},
                        SurfaceFunctionType<BarySurface, GenBarySurface>{},
                        SurfaceFunctionType<UVSurface, GenUVSurface>{});

    template<class Surface>
    static constexpr SurfaceFunc<Surface, TriangleHit, TriData> GetSurfaceFunction()
    {
        using namespace PrimitiveSurfaceFind;
        return LoopAndFindType<Surface, SurfaceFunc<Surface, TriangleHit, TriData>,
                               decltype(GeneratorFunctionList)>(std::move(GeneratorFunctionList));
    }
};

class GPUPrimitiveTriangle final
    : public GPUPrimitiveGroup<TriangleHit, TriData, DefaultLeaf,
                               TriangleSurfaceGenerator,
                               TriFunctions::Hit, 
                               TriFunctions::Leaf, TriFunctions::AABB, 
                               TriFunctions::Area, TriFunctions::Center>
{
    public:
        static constexpr const char*            TypeName() { return "Triangle"; }

        static constexpr PrimitiveDataLayout    POS_LAYOUT = PrimitiveDataLayout::FLOAT_3;
        static constexpr PrimitiveDataLayout    UV_LAYOUT = PrimitiveDataLayout::FLOAT_2;
        static constexpr PrimitiveDataLayout    NORMAL_LAYOUT = PrimitiveDataLayout::FLOAT_3;
        static constexpr PrimitiveDataLayout    TANGENT_LAYOUT = PrimitiveDataLayout::FLOAT_3;
        static constexpr PrimitiveDataLayout    INDEX_LAYOUT = PrimitiveDataLayout::UINT64_1;

        static constexpr const char*            CULL_FLAG_NAME = "cullFace";

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