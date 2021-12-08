#pragma once

// Materials
#include "DebugMaterials.cuh"
#include "SimpleMaterials.cuh"
#include "UnrealMaterial.cuh"
#include "LambertMaterial.cuh"
#include "EmptyMaterial.cuh"
// Primitives
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"
#include "GPUPrimitiveEmpty.h"
// Lights
#include "GPULightPrimitive.cuh"
#include "GPULightSkySphere.cuh"
#include "GPULightPoint.cuh"
#include "GPULightDirectional.cuh"
#include "GPULightSpot.cuh"
#include "GPULightDisk.cuh"
#include "GPULightRectangular.cuh"
#include "GPULightNull.cuh"
// Misc
#include "PathTracerKC.cuh"
#include "WorkPool.h"
#include "GPUWork.cuh"
#include "RayAuxStruct.cuh"

template<class EGroup>
class PTBoundaryWork final
    : public GPUBoundaryWorkBatch<PathTracerGlobalState,
                                  PathTracerLocalState, RayAuxPath,
                                  EGroup, PathTracerBoundaryWork<EGroup>>
{
    private:
        bool                neeOn;
        bool                misOn;

        using Base = GPUBoundaryWorkBatch<PathTracerGlobalState,
                                          PathTracerLocalState, RayAuxPath,
                                          EGroup, PathTracerBoundaryWork<EGroup>>;

    protected:
    public:
        // Constructors & Destructor
                                PTBoundaryWork(const CPUEndpointGroupI& eg,
                                               const GPUTransformI* const* t,
                                               bool neeOn, bool misOn);
                                ~PTBoundaryWork() = default;

        void                    GetReady() override {}

        const char*             Type() const override { return Base::TypeName(); }
};

template<class MGroup, class PGroup>
class PTPathWork final
    : public GPUWorkBatch<PathTracerGlobalState,
                          PathTracerLocalState, RayAuxPath,
                          MGroup, PGroup, PathTracerPathWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
        bool                neeOn;
        bool                misOn;

        using Base = GPUWorkBatch<PathTracerGlobalState,
                                  PathTracerLocalState, RayAuxPath,
                                  MGroup, PGroup, PathTracerPathWork<MGroup>,
                                  PGroup::GetSurfaceFunction>;
    protected:
    public:
        // Constructors & Destructor
                            PTPathWork(const GPUMaterialGroupI& mg,
                                        const GPUPrimitiveGroupI& pg,
                                        const GPUTransformI* const* t,
                                        bool neeOn, bool misOn);
                            ~PTPathWork() = default;

        void                GetReady() override {}
        uint8_t             OutRayCount() const override;

        const char*         Type() const override { return Base::TypeName(); }
};

template<class E>
PTBoundaryWork<E>::PTBoundaryWork(const CPUEndpointGroupI& eg,
                                  const GPUTransformI* const* t,
                                  bool neeOn, bool misOn)
    : Base(eg, t)
    , neeOn(neeOn)
    , misOn(neeOn && misOn)
{}

template<class M, class P>
PTPathWork<M, P>::PTPathWork(const GPUMaterialGroupI& mg,
                             const GPUPrimitiveGroupI& pg,
                             const GPUTransformI* const* t,
                             bool neeOn, bool misOn)
    : GPUWorkBatch<PathTracerGlobalState,
                   PathTracerLocalState, RayAuxPath,
                   M, P, PathTracerPathWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
    , neeOn(neeOn)
    , misOn(neeOn && misOn)
{
    // Populate localData
    this->localData.emptyPrimitive = false;
}

template<class M, class P>
uint8_t PTPathWork<M, P>::OutRayCount() const
{
    // Incorporate NEE Ray as an addition
    // If material can be sampled (i.e no Dirac Delta BxDF)
    if(!this->materialGroup.CanBeSampled())
    {
        // Material Cannot be sampled just allocate whatever
        // the material is requesting
        return this->materialGroup.SampleStrategyCount();
    }
    else if(this->materialGroup.SampleStrategyCount() != 0)
    {
        // Material can be sampled
        // Add one extra nee ray allocation
        uint8_t neeRay = (neeOn) ? 1 : 0;
        // Do not allocate any extra rays for MIS Ray
        // We will use path ray as MIS ray
        return this->materialGroup.SampleStrategyCount() + neeRay;
    }
    // Material does not require any samples meaning it is boundary material
    // Do not allocate any rays for this kind of material
    return 0;
}

// Path Tracer Work Batches
// ===================================================
// Boundary
extern template class PTBoundaryWork<CPULightGroupNull>;
extern template class PTBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>;
extern template class PTBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>;
extern template class PTBoundaryWork<CPULightGroupSkySphere>;
extern template class PTBoundaryWork<CPULightGroupPoint>;
extern template class PTBoundaryWork<CPULightGroupDirectional>;
extern template class PTBoundaryWork<CPULightGroupSpot>;
extern template class PTBoundaryWork<CPULightGroupDisk>;
extern template class PTBoundaryWork<CPULightGroupRectangular>;
// ===================================================
// Path
extern template class PTPathWork<LambertCMat, GPUPrimitiveTriangle>;
extern template class PTPathWork<LambertCMat, GPUPrimitiveSphere>;

extern template class PTPathWork<ReflectMat, GPUPrimitiveTriangle>;
extern template class PTPathWork<ReflectMat, GPUPrimitiveSphere>;

extern template class PTPathWork<RefractMat, GPUPrimitiveTriangle>;
extern template class PTPathWork<RefractMat, GPUPrimitiveSphere>;

extern template class PTPathWork<LambertMat, GPUPrimitiveTriangle>;
extern template class PTPathWork<LambertMat, GPUPrimitiveSphere>;

extern template class PTPathWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class PTPathWork<UnrealMat, GPUPrimitiveSphere>;
// ===================================================
using PTBoundaryWorkerList = TypeList<PTBoundaryWork<CPULightGroupNull>,
                                      PTBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>,
                                      PTBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>,
                                      PTBoundaryWork<CPULightGroupSkySphere>,
                                      PTBoundaryWork<CPULightGroupPoint>,
                                      PTBoundaryWork<CPULightGroupDirectional>,
                                      PTBoundaryWork<CPULightGroupSpot>,
                                      PTBoundaryWork<CPULightGroupDisk>,
                                      PTBoundaryWork<CPULightGroupRectangular>>;
// ===================================================
using PTPathWorkerList = TypeList<PTPathWork<LambertCMat, GPUPrimitiveTriangle>,
                                  PTPathWork<LambertCMat, GPUPrimitiveSphere>,
                                  PTPathWork<ReflectMat, GPUPrimitiveTriangle>,
                                  PTPathWork<ReflectMat, GPUPrimitiveSphere>,
                                  PTPathWork<RefractMat, GPUPrimitiveTriangle>,
                                  PTPathWork<RefractMat, GPUPrimitiveSphere>,
                                  PTPathWork<LambertMat, GPUPrimitiveTriangle>,
                                  PTPathWork<LambertMat, GPUPrimitiveSphere>,
                                  PTPathWork<UnrealMat, GPUPrimitiveTriangle>,
                                  PTPathWork<UnrealMat, GPUPrimitiveSphere>>;