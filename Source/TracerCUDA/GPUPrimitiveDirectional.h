#pragma once


#include "RayLib/Vector.h"
#include "RayLib/Quaternion.h"

#include "Random.cuh"
#include "GPUPrimitiveP.cuh"
#include "DefaultLeaf.h"
#include "GPUTransformI.h"
#include "GPUSurface.h"
#include "DeviceMemory.h"
#include "TypeTraits.h"

// Triangle Memory Layout
struct DirectionalData
{
    const Vector3f* directions;     // Direction
    const AABB3f*   spans;          // local space span
    const float*    distances;      // Starting distance of the direction
};

struct DirectionalHit {};

struct DirectionalFunctions
{
    __device__ __forceinline__
    static Vector3f Sample(// Output
                           Vector3f& normal,
                           float& pdf,
                           // Input
                           PrimitiveId primitiveId,
                           const DirectionalData& primData,
                           // I-O
                           RandomGPU& rng)
    {
        Vector3f direction = primData.directions[primitiveId];
        //Vector3f center = primData.spans[primitiveId].Centroid();
        //float area = Area(primitiveId, primData);
        // TODO: implement
        // Assume a disk ??
        
        normal = direction;        
        return Zero3f;
    }

    __device__ __forceinline__
    static void PDF(// Outputs
                    Vector3f& normal,
                    float& pdf,
                    float& distance,
                    // Inputs
                    const Vector3f& position,
                    const Vector3f& direction,
                    const GPUTransformI& transform,
                    const PrimitiveId primitiveId,
                    const DirectionalData& primData)
    {
        pdf = 1.0f;
        distance = primData.distances[primitiveId];
        normal = primData.directions[primitiveId];
    }

    // Triangle Hit Acceptance
    __device__ __forceinline__
    static HitResult Hit(// Output
                         HitKey& newMat,
                         PrimitiveId& newPrim,
                         DirectionalHit& newHit,
                         // I-O
                         RayReg& rayData,
                         // Input
                         const GPUTransformI& transform,
                         const DefaultLeaf& leaf,
                         const DirectionalData& primData)
    {
        //const AABB3f& span = primData.span[leaf.primitiveId];
        //// Convert Ray to local Space
        //RayF r = transform.WorldToLocal(rayData.ray);        
        //bool intersectsAABB = r.IntersectsAABB(span.Min(),
        //                                       span.Max(),
        //                                       Vector2f(rayData.tMin, 
        //                                                rayData.tMax));
        //// Check Alignment
        //const Vector3f& dir = primData.directions[leaf.primitiveId];
        //bool alignedDirections = (abs(r.getDirection().Dot(dir)) <= MathConstants::Epsilon);
        //return HitResult{false, alignedDirections && intersectsAABB};

        // Above code is highly unlikely and it will produce numeric errors
        // so just return false
        return HitResult{false, false};
    }

    __device__ __forceinline__
    static AABB3f AABB(const GPUTransformI& transform,
                       //
                       PrimitiveId primitiveId,
                       const DirectionalData& primData)
    {
        return transform.LocalToWorld(primData.spans[primitiveId]);
    }

    __device__ __forceinline__
    static float Area(PrimitiveId primitiveId, const DirectionalData& primData)
    {
        // Projected area
        const Vector3f& span = primData.spans[primitiveId].Span();
        const Vector3f& dir = primData.directions[primitiveId];
        Vector3f area = (Vector3(span[1], span[2], span[0]) *
                         Vector3(span[2], span[0], span[1]));
        return dir.Abs().Dot(area);
    }

    __device__ __forceinline__
    static Vector3 Center(const GPUTransformI& transform,
                          PrimitiveId primitiveId, const DirectionalData& primData)
    {        
        AABB3f worldSpan = transform.LocalToWorld(primData.spans[primitiveId]);
        return worldSpan.Centroid();
    }

    static constexpr auto Leaf = GenerateDefaultLeaf<DirectionalData>;
};

struct DirectionalSurfaceGenerator
{

    template <class Surface, SurfaceFunc<Surface, DirectionalHit, DirectionalData> SF>
    struct SurfaceFunctionType
    {
        using type = Surface;
        static constexpr auto SurfaceGeneratorFunction = SF;
    };

    static constexpr auto GeneratorFunctionList =
        std::make_tuple(SurfaceFunctionType<EmptySurface, 
                                            GenEmptySurface<DirectionalHit, DirectionalData>>{});

    template<class Surface>
    static constexpr SurfaceFunc<Surface, DirectionalHit, DirectionalData> GetSurfaceFunction()
    {
        using namespace PrimitiveSurfaceFind;
        return LoopAndFindType<Surface, SurfaceFunc<Surface, DirectionalHit, DirectionalData>,
                               decltype(GeneratorFunctionList)>(std::move(GeneratorFunctionList));
    }
};

class GPUPrimitiveDirectional final
    : public GPUPrimitiveGroup<DirectionalHit, DirectionalData, DefaultLeaf,
                               DirectionalSurfaceGenerator,
                               DirectionalFunctions::Hit,
                               DirectionalFunctions::Leaf, 
                               DirectionalFunctions::AABB,
                               DirectionalFunctions::Area, 
                               DirectionalFunctions::Center,
                               DirectionalFunctions::Sample,
                               DirectionalFunctions::PDF>
{
    public:
        static constexpr const char*    TypeName() { return "Directional"; }

        static constexpr const char*    NAME_DIRECTION = "direction";
        static constexpr const char*    NAME_SPAN_MIN = "spanMin";
        static constexpr const char*    NAME_SPAN_MAX = "spanMax";
        static constexpr const char*    NAME_DISTANCE = "distance";

    private:
        DeviceMemory                            memory;
        // List of ranges for each batch
        uint64_t                                totalPrimitiveCount;       
        // Misc Data
        std::map<uint32_t, Vector2ul>           batchRanges;
        std::map<uint32_t, AABB3>               batchAABBs;

    protected:
    public:
        // Constructors & Destructor
                                                GPUPrimitiveDirectional();
                                                ~GPUPrimitiveDirectional() = default;

        // Interface
        // Pirmitive type is used for delegating scene info to this class
        const char*                             Type() const override;
        // Allocates and Generates Data
        SceneError                              InitializeGroup(const NodeListing& surfaceDataNodes, double time,
                                                                const SurfaceLoaderGeneratorI& loaderGen,
                                                                const TextureNodeMap& textureNodes,
                                                                const std::string& scenePath) override;
        SceneError                              ChangeTime(const NodeListing& surfaceDatalNodes, double time,
                                                           const SurfaceLoaderGeneratorI& loaderGen,
                                                           const std::string& scenePath) override;
        // Access primitive range from Id
        Vector2ul                               PrimitiveBatchRange(uint32_t surfaceDataId) const override;
        AABB3                                   PrimitiveBatchAABB(uint32_t surfaceDataId) const override;

        // Primitive Transform Info for accelerator
        PrimTransformType                       TransformType() const override;
        // If primitive (by definition) is intersectable or not
        bool                                    IsIntersectable() const override;
        // Error check
        // Queries in order to check if this primitive group supports certain primitive data
        // Material may need that data
        bool                                    CanGenerateData(const std::string& s) const override;
};

static_assert(IsTracerClass<GPUPrimitiveDirectional>::value,
              "GPUPrimitiveDirectional is not a Tracer Class.");