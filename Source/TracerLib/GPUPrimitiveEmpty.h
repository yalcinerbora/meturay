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

// Triangle Hit Acceptance
__device__ __host__
inline HitResult EmptyClosestHit(// Output
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
inline AABB3f GenerateAABBEmpty(PrimitiveId primitiveId, const EmptyData& primData)
{
    Vector3f minInf(-INFINITY);
    return AABB3f(minInf, minInf);
}

__device__ __host__
inline float GenerateAreaEmpty(PrimitiveId primitiveId, const EmptyData& primData)
{
    return 0.0f;
}

__device__ __host__
inline Vector3f GenerateCenterEmpty(PrimitiveId primitiveId, const EmptyData& primData)
{
    return Zero3;
}

class GPUPrimitiveEmpty final
    : public GPUPrimitiveGroup<EmptyHit, EmptyData, EmptyLeaf,
                               EmptyClosestHit, GenerateEmptyLeaf,
                               GenerateAABBEmpty, GenerateAreaEmpty,
                               GenerateCenterEmpty>
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
        SceneError                              GenerateEstimatorInfo(std::vector<EstimatorInfo>&,
                                                                      HitKey key,
                                                                      uint32_t surfaceDataId) const override;
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