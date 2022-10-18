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
                                       const HitStructPtr gHitStructs) const = 0;
};

// Every Work Group is responsible to maintain this struct
template<class PrimGroup, class MatGroup>
class GPUMetaSurfaceGenerator
{
    public:
    using HitData   = typename PrimGroup::HitData;
    using PrimData  = typename PrimGroup::PrimitiveData;
    using MatData   = typename MatGroup::PrimitiveData;
    using SF        = SurfaceFunc<UVSurface,
                                  typename PrimGroup::HitData,
                                  typename PrimGroup::PrimitiveData>;
    // Surface Generator Function
    // Get UV surface, if primitive does not support uv surface
    // this will compile time fail
    // TODO: Compile time determine non uv surface zero out the uv etc
    static constexpr SF SurfFunc = PrimGroup::GetSurfaceFunction();

    private:
    // Single Pointers (Struct of Arrays)
    const PrimData&         gPrimData;
    const GPUMaterialI**    gLocalMaterials;
    const GPUTransformI**   gTransforms;

    __host__                GPUMetaSurfaceGenerator(const PrimData& gPData,
                                                    const GPUMaterialI** gLocalMaterials,
                                                    const GPUTransformI** gTransforms);
    __device__
    GPUMetaSurface          AcquireWork(// Ids
                                        uint32_t rayId,
                                        TransformId tId,
                                        PrimitiveId primId,
                                        HitKey workId,
                                        //This may change every frame
                                        // Thus provided as an argument
                                        const HitStructPtr gHitStructs) override;
};

class GPUMetaSurfaceGeneratorGroup
{
    private:
    const GPUMetaSurfaceGeneratorI**    gGeneratorInterfaces;
    const HitStructPtr                  gCurrentHitStructListPtr;

    public:
    // Constructors & Destructor
    __host__
                    GPUMetaSurfaceGeneratorGroup(const GPUMetaSurfaceGeneratorI**,
                                                 const HitStructPtr);

    __device__
    GPUMetaSurface  AcquireWork(uint32_t rayId,
                                TransformId tId,
                                PrimitiveId primId,
                                HitKey workId);
};

class GPUMetaSurfaceHandler
{
    private:
    const GPUMetaSurfaceGeneratorI** gGeneratorInterfaces;
    DeviceMemory                     genericMem;

    public:
                                    GPUMetaSurfaceHandler();

    TracerError                     Initialize(const GPUSceneI& scene);
    GPUMetaSurfaceGeneratorGroup    GetMetaSurfaceGroup(const HitStructPtr currentHitStructs);
};

template<class P, class M>
__host__ inline
GPUMetaSurfaceGenerator<P,M>::GPUMetaSurfaceGenerator(const PrimData& gPData,
                                                      const GPUMaterialI** gLocalMaterials,
                                                      const GPUTransformI** gTransforms)
    : gPrimData(gPData)
    , gLocalMaterials(gLocalMaterials)
    , gTransforms(gTransforms)
{}

template<class P, class M>
__device__ inline
GPUMetaSurface GPUMetaSurfaceGenerator<P,M>::AcquireWork(// Ids
                                                         uint32_t rayId,
                                                         TransformId tId,
                                                         PrimitiveId primId,
                                                         HitKey workKey,
                                                         //This may change every frame
                                                         // Thus provided as an argument
                                                         const HitStructPtr gHitStructs)
{
    // Find the material interface
    HitKey::Type workId = HitKey::FetchIdPortion(workKey);
    const GPUMaterialI* gMaterial = gLocalMaterials[workId];
    // Get Transform for surface generation and
    const GPUTransformI& transform = *gTransforms[tId];
    // Get hit data from the global array
    const HitData hit = gHitStructs.Ref<HitData>(rayId);
    UVSurface uvSurface = SurfFunc(hit, transform,
                                   ray.ray.getDirection(),
                                   primitiveId, primData);
    // Return the class
    return GPUMetaSurface(transform,
                          uvSurface,
                          gMaterial);
}

__host__ inline
GPUMetaSurfaceGeneratorGroup::GPUMetaSurfaceGeneratorGroup(const GPUMetaSurfaceGeneratorI** gGenPtrs,
                                                           const HitStructPtr gHitStructPtr)
    : gGeneratorInterfaces(gGenPtrs)
    , gCurrentHitStructListPtr(gHitStructPtr)
{}

__device__ inline
GPUMetaSurface GPUMetaSurfaceGeneratorGroup::AcquireWork(uint32_t rayId,
                                                         TransformId tId,
                                                         PrimitiveId primId,
                                                         HitKey workId)
{
    HitKey::Type workBatchId = HitKey::FetchBatchPortion(workId);
    const GPUMetaSurfaceGeneratorI* metaSurfGen = gGeneratorInterfaces[workBatchId];
    return metaSurfGen->AcquireWork(rayId, tId, primId, workId, gCurrentHitStructListPtr);
}

inline GPUMetaSurfaceHandler::GPUMetaSurfaceHandler()
    : gGeneratorInterfaces(nullptr)
{}

inline GPUMetaSurfaceGeneratorGroup GPUMetaSurfaceHandler::GetMetaSurfaceGroup(const HitStructPtr dCurrentHitStructs)
{
    return GPUMetaSurfaceGeneratorGroup(gGeneratorInterfaces, dCurrentHitStructs);
}