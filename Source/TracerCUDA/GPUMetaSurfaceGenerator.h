#pragma once

#include "GPUMetaSurface.h"
#include "DeviceMemory.h"
#include "RayLib/TracerError.h"
#include "GPUPrimitiveP.cuh"

class GPUSceneI;

// Implementation of these interfaces reside
// on the work groups
class GPUMetaSurfaceGeneratorI
{
    public:
    virtual ~GPUMetaSurfaceGeneratorI() = default;
    // Only Interface
    // Generates a Primitive Type / Material Type Agnostic
    // class that can be used to shade and acquire surface information
    // from another non work related kernel
    __device__
    virtual GPUMetaSurface AcquireWork(// Rest is ID
                                       uint32_t rayId,
                                       TransformId tId,
                                       PrimitiveId primId,
                                       HitKey workId,
                                       //This may change every frame
                                       // Thus provided as an argument
                                       const HitStructPtr gHitStructs,
                                       const RayGMem* gRaysIn) const = 0;
};

// Every Work Group is responsible to maintain this struct
template<class PrimGroup, class MatGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PrimGroup::HitData,
                              typename PrimGroup::PrimitiveData> SGen>
class GPUMetaSurfaceGenerator : public GPUMetaSurfaceGeneratorI
{
    public:
    using HitData   = typename PrimGroup::HitData;
    using PrimData  = typename PrimGroup::PrimitiveData;
    using MatData   = typename MatGroup::Data;
    using SF        = SurfaceFunc<UVSurface,
                                  typename PrimGroup::HitData,
                                  typename PrimGroup::PrimitiveData>;
    // Surface Generator Function
    // Get UV surface, if primitive does not support uv surface
    // this will compile time fail
    // TODO: Compile time determine non uv surface zero out the uv etc
    static constexpr SF SurfFunc = SGen();

    private:
    // Single Pointers (Struct of Arrays)
    const PrimData&             gPrimData;
    const GPUMaterialI**        gLocalMaterials;
    const GPUTransformI* const* gTransforms;

    public:
    // Constructors & Destructor
    __device__              GPUMetaSurfaceGenerator(const PrimData& gPData,
                                                    const GPUMaterialI** gLocalMaterials,
                                                    const GPUTransformI* const* gTransforms);

    __device__
    GPUMetaSurface          AcquireWork(// Rest is ID
                                        uint32_t rayId,
                                        TransformId tId,
                                        PrimitiveId primId,
                                        HitKey workId,
                                        //This may change every frame
                                        // Thus provided as an argument
                                        const HitStructPtr gHitStructs,
                                        const RayGMem* gRaysIn) const override;
};

template <class EndpointGroup>
class GPUBoundaryMetaSurfaceGenerator : public GPUMetaSurfaceGeneratorI
{
    public:
    using GPUEndpointType   = typename EndpointGroup::GPUType;
    using PrimData          = typename EndpointGroup::PrimitiveData;
    using HitData           = typename EndpointGroup::HitData;

    private:
    // Single Pointers (Struct of Arrays)
    const PrimData&             gPrimData;
    const GPUEndpointType*      gLocalLightInterfaces;
    const GPUTransformI* const* gTransforms;

    using SF        = SurfaceFunc<UVSurface,
                                  typename EndpointGroup::HitData,
                                  typename EndpointGroup::PrimitiveData>;
    // Surface Generator Function
    // TODO: lights (primitive backed) only supports UV surface
    // so this is fine currently, but it may be changed later
    static constexpr SF SurfFunc = EndpointGroup::SurfF;

    public:
    // Constructors & Destructor
    __device__              GPUBoundaryMetaSurfaceGenerator(const PrimData& gPData,
                                                            const GPUEndpointType* gLocalLightInterfaces,
                                                            const GPUTransformI* const* gTransforms);

    __device__
    GPUMetaSurface          AcquireWork(// Rest is ID
                                        uint32_t rayId,
                                        TransformId tId,
                                        PrimitiveId primId,
                                        HitKey workId,
                                        //This may change every frame
                                        // Thus provided as an argument
                                        const HitStructPtr gHitStructs,
                                        const RayGMem* gRaysIn) const override;
};

class GPUMetaSurfaceGeneratorGroup
{
    private:
    const GPUMetaSurfaceGeneratorI**    gGeneratorInterfaces;
    // Unsorted Current Ray Related Memory
    const RayGMem*                      gRaysIn;
    const HitKey*                       gWorkKeys;
    const PrimitiveId*                  gPrimIds;
    const TransformId*                  gTransformIds;
    const HitStructPtr                  gHitStructPtr;

    public:
    // Constructors & Destructor
    __host__        GPUMetaSurfaceGeneratorGroup(const GPUMetaSurfaceGeneratorI**,
                                                 const RayGMem* dRaysIn,
                                                 const HitKey* dWorkKeys,
                                                 const PrimitiveId* dPrimIds,
                                                 const TransformId* dTransformIds,
                                                 const HitStructPtr dHitStructPtr);

    __device__
    GPUMetaSurface  AcquireWork(uint32_t rayId) const;
    __device__
    PrimitiveId     PrimId(uint32_t rayId) const;
    __device__
    HitKey          WorkKey(uint32_t rayId) const;
    __device__
    TransformId     TransId(uint32_t rayId) const;
    __device__
    RayReg          Ray(uint32_t rayId) const;
};

class GPUMetaSurfaceHandler
{
    private:
    const GPUMetaSurfaceGeneratorI**    dGeneratorInterfaces;
    DeviceMemory                        generatorMem;

    public:
    // Constructors & Destructor
                                        GPUMetaSurfaceHandler();

    TracerError                         Initialize(const GPUSceneI& scene,
                                                   const WorkBatchMap& sceneWorkBatches);
    GPUMetaSurfaceGeneratorGroup        GetMetaSurfaceGroup(const RayGMem* dRaysIn,
                                                            const HitKey* dWorkKeys,
                                                            const PrimitiveId* dPrimIds,
                                                            const TransformId* dTransformIds,
                                                            const HitStructPtr dHitStructPtr);
};

template<class PrimGroup, class MatGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PrimGroup::HitData,
                              typename PrimGroup::PrimitiveData> SGen>
__device__ inline
GPUMetaSurfaceGenerator<PrimGroup, MatGroup, SGen>::GPUMetaSurfaceGenerator(const PrimData& gPData,
                                                                            const GPUMaterialI** gLocalMaterials,
                                                                            const GPUTransformI* const* gTransforms)
    : gPrimData(gPData)
    , gLocalMaterials(gLocalMaterials)
    , gTransforms(gTransforms)
{}

template<class PrimGroup, class MatGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PrimGroup::HitData,
                              typename PrimGroup::PrimitiveData> SGen>
__device__ inline
GPUMetaSurface GPUMetaSurfaceGenerator<PrimGroup, MatGroup, SGen>::AcquireWork(// Ids
                                                                               uint32_t rayId,
                                                                               TransformId tId,
                                                                               PrimitiveId primId,
                                                                               HitKey workKey,
                                                                               //This may change every frame
                                                                               // Thus provided as an argument
                                                                               const HitStructPtr gHitStructs,
                                                                               const RayGMem* gRaysIn) const
{
    const RayReg ray(gRaysIn, rayId);
    // Find the material interface
    HitKey::Type workId = HitKey::FetchIdPortion(workKey);
    const GPUMaterialI* gMaterial = gLocalMaterials[workId];
    // Get Transform for surface generation and
    const GPUTransformI* transform = gTransforms[tId];
    // Get hit data from the global array
    const HitData hit = gHitStructs.Ref<HitData>(rayId);
    UVSurface uvSurface = SurfFunc(hit, *transform,
                                   ray.ray.getDirection(),
                                   primId, gPrimData);
    // Return the class
    return GPUMetaSurface(transform,
                          uvSurface,
                          gMaterial);
}

template<class EndpointGroup>
__device__ inline
GPUBoundaryMetaSurfaceGenerator<EndpointGroup>::GPUBoundaryMetaSurfaceGenerator(const PrimData& gPData,
                                                                                const GPUEndpointType* gLocalLightInterfaces,
                                                                                const GPUTransformI* const* gTransforms)
    : gPrimData(gPData)
    , gLocalLightInterfaces(gLocalLightInterfaces)
    , gTransforms(gTransforms)
{}

template <class EndpointGroup>
__device__ inline
GPUMetaSurface GPUBoundaryMetaSurfaceGenerator<EndpointGroup>::AcquireWork(// Rest is ID
                                                                           uint32_t rayId,
                                                                           TransformId tId,
                                                                           PrimitiveId primId,
                                                                           HitKey workKey,
                                                                           //This may change every frame
                                                                           // Thus provided as an argument
                                                                           const HitStructPtr gHitStructs,
                                                                           const RayGMem* gRaysIn) const
{
    const RayReg ray(gRaysIn, rayId);
    // Find the material interface
    HitKey::Type workId = HitKey::FetchIdPortion(workKey);
    const GPULightI* gLightI = (gLocalLightInterfaces) ? gLocalLightInterfaces + workId : nullptr;
    // Get Transform for surface generation and
    const GPUTransformI* transform = gTransforms[tId];
    // Get hit data from the global array
    const HitData hit = gHitStructs.Ref<HitData>(rayId);
    // Somehow we missed the light,
    bool primLight = (gLightI && gLightI->IsPrimitiveBackedLight());
    bool primLightButMissed = (primLight && primId >= INVALID_PRIMITIVE_ID);
    if(primLightButMissed)
    {
        return GPUMetaSurface(transform,
                              UVSurface{},
                              gLightI);
    }
    else
    {
        UVSurface uvSurface = SurfFunc(hit, *transform,
                                       ray.ray.getDirection(),
                                       primId, gPrimData);
        return GPUMetaSurface(transform,
                              uvSurface,
                              gLightI);
    }
}

__host__ inline
GPUMetaSurfaceGeneratorGroup::GPUMetaSurfaceGeneratorGroup(const GPUMetaSurfaceGeneratorI** gGenPtrs,
                                                           const RayGMem* dRaysIn,
                                                           const HitKey* dWorkKeys,
                                                           const PrimitiveId* dPrimIds,
                                                           const TransformId* dTransformIds,
                                                           const HitStructPtr dHitStructPtr)
    : gGeneratorInterfaces(gGenPtrs)
    , gRaysIn(dRaysIn)
    , gWorkKeys(dWorkKeys)
    , gPrimIds(dPrimIds)
    , gTransformIds(dTransformIds)
    , gHitStructPtr(dHitStructPtr)
{}

__device__ inline
GPUMetaSurface GPUMetaSurfaceGeneratorGroup::AcquireWork(uint32_t rayId) const
{
    TransformId tId = gTransformIds[rayId];
    PrimitiveId pId = gPrimIds[rayId];
    HitKey workId = gWorkKeys[rayId];

    HitKey::Type workBatchId = HitKey::FetchBatchPortion(workId);
    if(workBatchId == HitKey::NullBatch)
    {
        printf("NULLBATCH!\n");
        //__trap();
        return GPUMetaSurface(nullptr, UVSurface{},
                              static_cast<GPUMaterialI*>(nullptr));
    }

    const GPUMetaSurfaceGeneratorI* gMetaSurfGen = gGeneratorInterfaces[workBatchId];
    return gMetaSurfGen->AcquireWork(rayId, tId, pId, workId, gHitStructPtr, gRaysIn);
}

__device__ inline
PrimitiveId GPUMetaSurfaceGeneratorGroup::PrimId(uint32_t rayId) const
{
    return gPrimIds[rayId];
}

__device__ inline
HitKey GPUMetaSurfaceGeneratorGroup::WorkKey(uint32_t rayId) const
{
    return gWorkKeys[rayId];
}

__device__ inline
TransformId GPUMetaSurfaceGeneratorGroup::TransId(uint32_t rayId) const
{
    return gTransformIds[rayId];
}

__device__ inline
RayReg GPUMetaSurfaceGeneratorGroup::Ray(uint32_t rayId) const
{
    return RayReg(gRaysIn, rayId);
}

inline GPUMetaSurfaceHandler::GPUMetaSurfaceHandler()
    : dGeneratorInterfaces(nullptr)
{}

inline GPUMetaSurfaceGeneratorGroup GPUMetaSurfaceHandler::GetMetaSurfaceGroup(const RayGMem* dRaysIn,
                                                                               const HitKey* dWorkKeys,
                                                                               const PrimitiveId* dPrimIds,
                                                                               const TransformId* dTransformIds,
                                                                               const HitStructPtr dHitStructPtr)
{
    return GPUMetaSurfaceGeneratorGroup(dGeneratorInterfaces, dRaysIn, dWorkKeys,
                                        dPrimIds, dTransformIds, dHitStructPtr);
}