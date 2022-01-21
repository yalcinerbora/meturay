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

#include "RNGenerator.h"
#include "GPUPrimitiveP.cuh"
#include "BinarySearch.cuh"

#include "GPUTransformI.h"
#include "GPUSurface.h"
#include "DefaultLeaf.h"
#include "DeviceMemory.h"
#include "TypeTraits.h"
#include "TextureFunctions.h"

class SurfaceDataLoaderI;
using SurfaceDataLoaders = std::vector<std::unique_ptr<SurfaceDataLoaderI>>;

// Triangle Memory Layout
struct TriData
{
    // Kinda Perf Hog but most memory efficient
    // Binary search cull face for each prim
    const GPUBitmap**   alphaMaps;
    const bool*         cullFace;
    const uint64_t*     primOffsets;
    uint32_t            primBatchCount;

    // Per vertex attributes
    const Vector3f*     positions;
    const QuatF*        tbnRotations;
    const Vector2*      uvs;
    // Single indexed vertices
    const uint64_t*     indexList;
};

// Triangle Hit is barycentric coordinates
// c is (1-a-b) thus it is not stored.
using TriangleHit = Vector2f;

struct TriFunctions
{
    __device__ inline
    static Vector3f SamplePosition(// Output
                                   Vector3f& normal,
                                   float& pdf,
                                   // Input
                                   const GPUTransformI& transform,
                                   //
                                   PrimitiveId primitiveId,
                                   const TriData& primData,
                                   // I-O
                                   RNGeneratorGPUI& rng)
    {
        Vector2f xi = rng.Uniform2D();
        float r1 = sqrt(xi[0]);
        float r2 = xi[1];
        // Generate Random Barycentrics
        // Osada 2002
        // http://graphics.stanford.edu/courses/cs468-08-fall/pdf/osada.pdf
        float a = 1 - r1;
        float b = (1 - r2) * r1;
        float c = r1 * r2;

        uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[primitiveId * 3 + 2];

        Vector3 position0 = primData.positions[index0];
        Vector3 position1 = primData.positions[index1];
        Vector3 position2 = primData.positions[index2];

        pdf = 1.0f / TriFunctions::Area(primitiveId, primData);

        // Calculate Normal
        // CCW
        //Vector3 vec0 = position1 - position0;
        //Vector3 vec1 = position2 - position0;
        //normal = Cross(vec0, vec1).Normalize();
        QuatF q0 = primData.tbnRotations[index0].Normalize();
        QuatF q1 = primData.tbnRotations[index1].Normalize();
        QuatF q2 = primData.tbnRotations[index2].Normalize();
        QuatF tbn = Quat::BarySLerp(q0, q1, q2, a, b);
        Vector3 Z_AXIS = ZAxis;

        normal = tbn.Conjugate().ApplyRotation(Z_AXIS);
        Vector3 position = (position0 * a +
                            position1 * b +
                            position2 * c);

        normal = transform.LocalToWorld(normal, true);
        position = transform.LocalToWorld(position);

        return position;
    }

    __device__ inline
    static void PositionPdfFromReference(// Outputs
                                         Vector3f& normal,
                                         float& pdf,
                                         float& distance,
                                         // Inputs
                                         const RayF& ray,
                                         const GPUTransformI& transform,
                                         //
                                         const PrimitiveId primitiveId,
                                         const TriData& primData)
    {
        // Find the primitive
        float index;
        GPUFunctions::BinarySearchInBetween(index, primitiveId,
                                            primData.primOffsets,
                                            primData.primBatchCount);
        uint32_t indexInt = static_cast<uint32_t>(index);
        bool cullBackface = primData.cullFace[indexInt];

        // Find the primitive
        uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[primitiveId * 3 + 2];

        Vector3 position0 = primData.positions[index0];
        Vector3 position1 = primData.positions[index1];
        Vector3 position2 = primData.positions[index2];

        RayF r = ray;
        r = transform.WorldToLocal(r);

        Vector3 baryCoords;
        bool intersects = r.IntersectsTriangle(baryCoords, distance,
                                               position0,
                                               position1,
                                               position2,
                                               cullBackface);

        // Check if an alpha map exists and accept/reject intersection
        const GPUBitmap* alphaMap = primData.alphaMaps[indexInt];
        if(alphaMap && intersects)
        {
            Vector2f uv0 = primData.uvs[index0];
            Vector2f uv1 = primData.uvs[index1];
            Vector2f uv2 = primData.uvs[index2];
            Vector2f uv = (baryCoords[0] * uv0 +
                           baryCoords[1] * uv1 +
                           baryCoords[2] * uv2);

            bool opaque = (*alphaMap)(uv);
            intersects &= opaque;
        }

        if(intersects)
        {
            QuatF q0 = primData.tbnRotations[index0].Normalize();
            QuatF q1 = primData.tbnRotations[index1].Normalize();
            QuatF q2 = primData.tbnRotations[index2].Normalize();
            QuatF tbn = Quat::BarySLerp(q0, q1, q2,
                                        baryCoords[0],
                                        baryCoords[1]);
            // Tangent Space to Local Space Transform
            Vector3 Z_AXIS = ZAxis;
            normal = tbn.Conjugate().ApplyRotation(Z_AXIS);
            // Local Space to World Space Transform
            normal = transform.LocalToWorld(normal);
        }

        // TODO: THIS IS WRONG?
        // Since alpha map can cull particular positions of the primitive
        // pdf is not uniform (thus it is not 1/Area)
        // fix it later since it is not common to lights having alpha mapped primitive

        if(intersects)
            pdf = 1.0f / TriFunctions::Area(primitiveId, primData);
        else pdf = 0.0f;
    }

    __device__ inline
    static float PositionPdfFromHit(// Inputs
                                    const Vector3f&,
                                    const Vector3f&,
                                    const QuatF&,
                                    //
                                    const PrimitiveId primitiveId,
                                    const TriData& primData)
    {
        return 1.0f / TriFunctions::Area(primitiveId, primData);
    }

    template <class GPUTransform>
    __device__ inline
    static bool IntersectsT(// Output
                            float& newT,
                            TriangleHit& newHit,
                            // I-O
                            const RayReg& rayData,
                            // Input
                            const GPUTransform& transform,
                            const DefaultLeaf& leaf,
                            const TriData& primData)
    {
        // Find the primitive
        float batchIndex;
        GPUFunctions::BinarySearchInBetween(batchIndex, leaf.primitiveId, primData.primOffsets, primData.primBatchCount);
        uint32_t batchIndexInt = static_cast<uint32_t>(batchIndex);

        const bool cullBackface = primData.cullFace[batchIndexInt];

        // Get Position
        uint64_t index0 = primData.indexList[leaf.primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[leaf.primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[leaf.primitiveId * 3 + 2];

        Vector3 position0 = primData.positions[index0];
        Vector3 position1 = primData.positions[index1];
        Vector3 position2 = primData.positions[index2];

        // Do Intersection test on local space
        RayF r = transform.WorldToLocal(rayData.ray);
        //
        float t;
        Vector3 baryCoords;
        bool intersects = r.IntersectsTriangle(baryCoords, t,
                                               position0,
                                               position1,
                                               position2,
                                               cullBackface);
        if(intersects)
        {
            newT = t;
            newHit = Vector2f(baryCoords[0], baryCoords[1]);
        }
        return intersects;
    }

    static constexpr auto& Intersects = IntersectsT<GPUTransformI>;

    __device__ inline
    static bool AlphaTest(// Input
                          const TriangleHit& potentialHit,
                          const DefaultLeaf& leaf,
                          const TriData& primData)
    {
        // Find the primitive
        float batchIndex;
        GPUFunctions::BinarySearchInBetween(batchIndex, leaf.primitiveId, primData.primOffsets, primData.primBatchCount);
        uint32_t batchIndexInt = static_cast<uint32_t>(batchIndex);
        const GPUBitmap* alphaMap = primData.alphaMaps[batchIndexInt];
        // Check if an alpha map does not exist
        // Accept intersection
        if(!alphaMap) return true;

        uint64_t index0 = primData.indexList[leaf.primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[leaf.primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[leaf.primitiveId * 3 + 2];

        Vector3 baryCoords = Vector3f(potentialHit[0],
                                      potentialHit[1],
                                      1.0f - potentialHit[1] - potentialHit[0]);

        Vector2f uv0 = primData.uvs[index0];
        Vector2f uv1 = primData.uvs[index1];
        Vector2f uv2 = primData.uvs[index2];

        Vector2f uv = (baryCoords[0] * uv0 +
                        baryCoords[1] * uv1 +
                        baryCoords[2] * uv2);

        return (*alphaMap)(uv);
    }

    __device__ inline
    static AABB3f AABB(const GPUTransformI& transform,
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

    __device__ inline
    static float Area(PrimitiveId primitiveId, const TriData& primData)
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

    __device__ inline
    static Vector3 Center(const GPUTransformI& transform,
                          PrimitiveId primitiveId, const TriData& primData)
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

        return (position0 * 0.33333f +
                position1 * 0.33333f +
                position2 * 0.33333f);
    }

    __device__ inline
    static void AcquirePositions(// Output
                                 Vector3f positions[3],
                                 // Inputs
                                 PrimitiveId primitiveId,
                                 const TriData& primData)
    {
        uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[primitiveId * 3 + 2];
        positions[0] = primData.positions[index0];
        positions[1] = primData.positions[index1];
        positions[2] = primData.positions[index2];
    }

    static constexpr auto& Leaf = GenerateDefaultLeaf<TriData>;
};

class GPUPrimitiveTriangle;

struct TriangleSurfaceGenerator
{
    __device__ inline
    static BasicSurface GenBasicSurface(const TriangleHit& baryCoords,
                                        const GPUTransformI& transform,
                                        //
                                        PrimitiveId primitiveId,
                                        const TriData& primData)
    {
        //float c = 1 - baryCoords[0] - baryCoords[1];

        uint64_t i0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t i1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t i2 = primData.indexList[primitiveId * 3 + 2];

        QuatF q0 = primData.tbnRotations[i0];//.Normalize();
        QuatF q1 = primData.tbnRotations[i1];//.Normalize();
        QuatF q2 = primData.tbnRotations[i2];//.Normalize();
        QuatF tbn = Quat::BarySLerp(q0, q1, q2,
                                    baryCoords[0],
                                    baryCoords[1]);
        tbn.NormalizeSelf();
        tbn = tbn * transform.ToLocalRotation();
        return BasicSurface{tbn};
    }

    __device__ inline
    static BarySurface GenBarySurface(const TriangleHit& baryCoords,
                                      const GPUTransformI&,
                                      //
                                      PrimitiveId,
                                      const TriData&)
    {
        float c = 1.0f - baryCoords[0] - baryCoords[1];
        return BarySurface{Vector3(baryCoords[0], baryCoords[1], c)};
    }

    __device__ inline
    static UVSurface GenUVSurface(const TriangleHit& baryCoords,
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
        std::make_tuple(SurfaceFunctionType<EmptySurface, DefaultGenEmptySurface<TriangleHit, TriData>>{},
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
                               TriangleSurfaceGenerator, TriFunctions,
                               PrimTransformType::CONSTANT_LOCAL_TRANSFORM,
                               3>
{
    public:
        static constexpr const char*            TypeName() { return "Triangle"; }

        static constexpr PrimitiveDataLayout    POS_LAYOUT = PrimitiveDataLayout::FLOAT_3;
        static constexpr PrimitiveDataLayout    UV_LAYOUT = PrimitiveDataLayout::FLOAT_2;
        static constexpr PrimitiveDataLayout    NORMAL_LAYOUT = PrimitiveDataLayout::FLOAT_3;
        static constexpr PrimitiveDataLayout    TANGENT_LAYOUT = PrimitiveDataLayout::FLOAT_3;
        static constexpr PrimitiveDataLayout    INDEX_LAYOUT = PrimitiveDataLayout::UINT64_1;

        static constexpr const char*            CULL_FLAG_NAME = "cullFace";
        static constexpr const char*            ALPHA_MAP_NAME = "alphaMap";

        using LoadedBitmapIndices = std::map<std::pair<uint32_t, TextureChannelType>, uint32_t>;

    private:
        DeviceMemory                            memory;
        // List of ranges for each batch
        uint64_t                                totalPrimitiveCount;
        uint64_t                                totalDataCount;
        // CPU Allocation of Bitmaps
        LoadedBitmapIndices                     loadedBitmaps;
        CPUBitmapGroup                          bitmaps;
        // Misc Data
        std::map<uint32_t, Vector2ul>           batchRanges;
        std::map<uint32_t, Vector2ul>           batchDataRanges;
        std::map<uint32_t, AABB3>               batchAABBs;
        std::map<uint32_t, bool>                batchAlphaMapFlag;
        std::map<uint32_t, bool>                batchBackFaceCullFlag;

    protected:
    public:
        // Constructors & Destructor
                                                GPUPrimitiveTriangle();
                                                ~GPUPrimitiveTriangle() = default;

        // Interface
        // Primitive type is used for delegating scene info to this class
        const char*                             Type() const override;
        // Allocates and Generates Data
        SceneError                              InitializeGroup(const NodeListing& surfaceDataNodes, double time,
                                                                const SurfaceLoaderGeneratorI&,
                                                                const TextureNodeMap& textureNodes,
                                                                const std::string& scenePath) override;
        SceneError                              ChangeTime(const NodeListing& surfaceDataNodes, double time,
                                                           const SurfaceLoaderGeneratorI&,
                                                           const std::string& scenePath) override;
        // Access primitive range from Id
        Vector2ul                               PrimitiveBatchRange(uint32_t surfaceDataId) const override;
        AABB3                                   PrimitiveBatchAABB(uint32_t surfaceDataId) const override;
        bool                                    PrimitiveBatchHasAlphaMap(uint32_t surfaceDataId) const override;
        bool                                    PrimitiveBatchBackFaceCulled(uint32_t surfaceDataId) const override;
        // Query
        // How many primitives are available on this class
        // This includes the indexed primitive count
        uint64_t                                TotalPrimitiveCount() const override;
        // Total primitive count but not indexed
        uint64_t                                TotalDataCount() const override;
        // Primitive Transform Info for accelerator
        bool                                    IsTriangle() const override;
        // Error check
        // Queries in order to check if this primitive group supports certain primitive data
        // Material may need that data
        bool                                    CanGenerateData(const std::string& s) const override;
};

static_assert(IsTracerClass<GPUPrimitiveTriangle>::value,
              "GPUPrimitiveTriangle is not a Tracer Class.");