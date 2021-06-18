#pragma once

#pragma once
/**

Default Triangle Implementation

Has three types of data
Position, Normal and UV.

All of them should be provided

*/

#include <map>
#include <type_traits>

#include "RayLib/Vector.h"

#include "DefaultLeaf.h"
#include "GPUPrimitiveP.cuh"
#include "TypeTraits.h"
#include "Random.cuh"
#include "GPUSurface.h"

struct EmptyData {};
struct EmptyHit {};

struct EPrimFunctions
{
    __device__ __forceinline__
    static Vector3f Sample(// Output
                           Vector3f& normal,
                           float& pdf,
                           // Input
                           PrimitiveId primitiveId,
                           const EmptyData& primData,
                           // I-O
                           RandomGPU& rng)
    {
        return Zero3;
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
                    const EmptyData& primData)
    {
        distance = INFINITY;
        pdf = 0.0f;
    }

    __device__
    static HitResult Hit(// Output
                         HitKey& newMat,
                         PrimitiveId& newPrimitive,
                         EmptyHit& newHit,
                         // I-O
                         RayReg& rayData,
                         // Input
                         const GPUTransformI& transform,
                         const EmptyLeaf& leaf,
                         const EmptyData& primData)
    {
        return HitResult{false, -FLT_MAX};
    }

    __device__ __forceinline__
    static AABB3f AABB(const GPUTransformI& transform,
                              PrimitiveId primitiveId, const EmptyData& primData)
    {
        Vector3f minInf(-INFINITY);
        return AABB3f(minInf, minInf);
    }

    __device__ __forceinline__
    static float Area(PrimitiveId primitiveId, const EmptyData& primData)
    {
        return 0.0f;
    }

    __device__ __forceinline__
    static Vector3f Center(const GPUTransformI& transform,
                           PrimitiveId primitiveId, const EmptyData& primData)
    {
        return Zero3;
    }

    static constexpr auto Leaf = GenerateEmptyLeaf<EmptyData>;
};

struct EmptySurfaceGenerator
{
    template <class Surface, SurfaceFunc<Surface, EmptyHit, EmptyData> SF>
    struct SurfaceFunctionType
    {
        using type = Surface;
        static constexpr auto SurfaceGeneratorFunction = SF;
    };

    static constexpr auto GeneratorFunctionList =
        std::make_tuple(SurfaceFunctionType<EmptySurface,
                                            GenEmptySurface<EmptyHit, EmptyData>>{},
                        SurfaceFunctionType<BasicSurface,
                                            GenBasicSurface<EmptyHit, EmptyData>>{});

    template<class Surface>
    static constexpr SurfaceFunc<Surface, EmptyHit, EmptyData> GetSurfaceFunction()
    {
        using namespace PrimitiveSurfaceFind;
        return LoopAndFindType<Surface, SurfaceFunc<Surface, EmptyHit, EmptyData>,
                               decltype(GeneratorFunctionList)>(std::move(GeneratorFunctionList));
    }
};

class GPUPrimitiveEmpty final
    : public GPUPrimitiveGroup<EmptyHit, EmptyData, EmptyLeaf,
                               EmptySurfaceGenerator,
                               EPrimFunctions::Hit,
                               EPrimFunctions::Leaf, EPrimFunctions::AABB,
                               EPrimFunctions::Area, EPrimFunctions::Center,
                               EPrimFunctions::Sample,
                               EPrimFunctions::PDF>
{
    public:
        static constexpr const char*            TypeName() { return BaseConstants::EMPTY_PRIMITIVE_NAME; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                                GPUPrimitiveEmpty();
                                                ~GPUPrimitiveEmpty() = default;

        // Interface
        // Pirmitive type is used for delegating scene info to this class
        const char*                             Type() const override;
        // Allocates and Generates Data
        SceneError                              InitializeGroup(const NodeListing& surfaceDatalNodes, double time,
                                                                const SurfaceLoaderGeneratorI&,
                                                                const TextureNodeMap&,
                                                                const std::string&) override;
        SceneError                              ChangeTime(const NodeListing& surfaceDatalNodes, double time,
                                                           const SurfaceLoaderGeneratorI&,
                                                           const std::string&) override;
        // Access primitive range from Id
        Vector2ul                               PrimitiveBatchRange(uint32_t surfaceDataId) const override;
        AABB3                                   PrimitiveBatchAABB(uint32_t surfaceDataId) const override;

        // Primitive Transform Info for accelerator
        PrimTransformType                       TransformType() const override;

        // Error check
        // Queries in order to check if this primitive group supports certain primitive data
        // Material may need that data
        bool                                    CanGenerateData(const std::string& s) const override;
};

static_assert(IsTracerClass<GPUPrimitiveEmpty>::value,
              "GPUPrimitiveEmpty is not a Tracer Class.");