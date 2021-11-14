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
    static constexpr auto& Hit                      = DefaultAcceptHit<EmptyHit, EmptyData, EmptyLeaf>;
    static constexpr auto& AABB                     = DefaultAABBGen<EmptyData>;
    static constexpr auto& Area                     = DefaultAreaGen<EmptyData>;
    static constexpr auto& Center                   = DefaultCenterGen<EmptyData>;
    static constexpr auto& SamplePosition           = DefaultSamplePos<EmptyData>;
    static constexpr auto& PositionPdfFromReference = DefaultPDFPosRef<EmptyData>;
    static constexpr auto& PositionPdfFromHit       = DefaultPDFPosHit<EmptyData>;
    static constexpr auto& Leaf                     = GenerateEmptyLeaf<EmptyData>;
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
                                            DefaultGenEmptySurface<EmptyHit, EmptyData>>{},
                        SurfaceFunctionType<BasicSurface,
                                            DefaultGenBasicSurface<EmptyHit, EmptyData>>{},
                        SurfaceFunctionType<UVSurface,
                                            DefaultGenUvSurface<EmptyHit, EmptyData>>{});

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
                               EPrimFunctions>
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