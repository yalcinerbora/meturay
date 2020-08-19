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

struct EmptyData {};
struct EmptyHit {};

struct EPrimFunctions
{

// Triangle Hit Acceptance
    __device__ __host__
    static inline HitResult Hit(// Output
                                HitKey& newMat,
                                PrimitiveId& newPrimitive,
                                EmptyHit& newHit,
                                // I-O
                                RayReg& rayData,
                                // Input
                                const EmptyLeaf& leaf,
                                const EmptyData& primData)
    {
        return HitResult{false, -FLT_MAX};
    }

    __device__ __host__
    static inline AABB3f AABB(PrimitiveId primitiveId, const EmptyData& primData)
    {
        Vector3f minInf(-INFINITY);
        return AABB3f(minInf, minInf);
    }

    __device__ __host__
    static inline float Area(PrimitiveId primitiveId, const EmptyData& primData)
    {
        return 0.0f;
    }

    __device__ __host__
    static inline Vector3f Center(PrimitiveId primitiveId, const EmptyData& primData)
    {
        return Zero3;
    }

    __device__ __host__
    static inline Matrix3x3 TSMatrix(const EmptyHit& hit,
                                     PrimitiveId,
                                     const EmptyData&)
    {
        return Indentity3x3;
    }

    static constexpr auto Leaf = GenerateEmptyLeaf<EmptyData>;
    static constexpr auto LocalToWorld = ToLocalSpace<EmptyData>;
    static constexpr auto WorldToLocal = FromLocalSpace<EmptyData>;
};

class GPUPrimitiveEmpty final
    : public GPUPrimitiveGroup<EmptyHit, EmptyData, EmptyLeaf,
                               EPrimFunctions::Hit, EPrimFunctions::Leaf,
                               EPrimFunctions::AABB, EPrimFunctions::Area,
                               EPrimFunctions::Center, EPrimFunctions::LocalToWorld,
                               EPrimFunctions::WorldToLocal, EPrimFunctions::TSMatrix>
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
                                                                const std::string&) override;
        SceneError                              ChangeTime(const NodeListing& surfaceDatalNodes, double time,
                                                           const SurfaceLoaderGeneratorI&,
                                                           const std::string&) override;
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

static_assert(IsTracerClass<GPUPrimitiveEmpty>::value, 
              "GPUPrimitiveEmpty is not a Tracer Class.");