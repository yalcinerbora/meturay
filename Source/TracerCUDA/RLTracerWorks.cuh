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
#include "RLTracerKC.cuh"
#include "WorkPool.h"
#include "GPUWork.cuh"
#include "RayAuxStruct.cuh"

template<class EGroup>
class RLBoundaryWork
    : public GPUBoundaryWorkBatch<RLTracerGlobalState,
                                  RLTracerLocalState, RayAuxRL,
                                  EGroup, RLTracerBoundaryWork<EGroup>>
{
    private:
        bool                            neeOn;
        bool                            misOn;

        using Base = GPUBoundaryWorkBatch<RLTracerGlobalState,
                                          RLTracerLocalState, RayAuxRL,
                                          EGroup, RLTracerBoundaryWork<EGroup>>;

    protected:
    public:
        // Constructors & Destructor
                                        RLBoundaryWork(const CPUEndpointGroupI& eg,
                                                       const GPUTransformI* const* t,
                                                       bool neeOn, bool misOn);
                                        ~RLBoundaryWork() = default;

        void                            GetReady() override {}

        const char*                     Type() const override { return Base::TypeName(); }
};

template<class MGroup, class PGroup>
class RLWork
    : public GPUWorkBatch<RLTracerGlobalState,
                          RLTracerLocalState, RayAuxRL,
                          MGroup, PGroup, RLTracerPathWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
        bool                            neeOn;
        bool                            misOn;

        using Base = GPUWorkBatch<RLTracerGlobalState,
                                  RLTracerLocalState, RayAuxRL,
                                  MGroup, PGroup, RLTracerPathWork<MGroup>,
                                  PGroup::GetSurfaceFunction>;

    protected:
    public:
        // Constructors & Destructor
                                        RLWork(const GPUMaterialGroupI& mg,
                                               const GPUPrimitiveGroupI& pg,
                                               const GPUTransformI* const* t,
                                               bool neeOn, bool misOn);
                                        ~RLWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override;

        const char*                     Type() const override { return Base::TypeName(); }
};

template<class E>
RLBoundaryWork<E>::RLBoundaryWork(const CPUEndpointGroupI& eg,
                                  const GPUTransformI* const* t,
                                  bool neeOn, bool misOn)
    : Base(eg, t)
    , neeOn(neeOn)
    , misOn(misOn)
{}

template<class M, class P>
RLWork<M, P>::RLWork(const GPUMaterialGroupI& mg,
                     const GPUPrimitiveGroupI& pg,
                     const GPUTransformI* const* t,
                     bool neeOn, bool misOn)
    : Base(mg, pg, t)
    , neeOn(neeOn)
    , misOn(misOn)
{
    // Populate localData
    this->localData.emptyPrimitive = false;
}

template<class M, class P>
uint8_t RLWork<M, P>::OutRayCount() const
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
        // Add one extra MIS ray for allocation
        // MIS ray is extra NEE ray for multiple importance sampling
        uint8_t misRay = (neeOn && misOn) ? 1 : 0;

        return this->materialGroup.SampleStrategyCount() + misRay + neeRay;
    }
    // Material does not require any samples meaning it is boundary material
    // Do not allocate any rays for this kind of material
    return 0;
}

// PPG Tracer Work Batches
// ===================================================
// Boundary
extern template class RLBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>;
extern template class RLBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>;
extern template class RLBoundaryWork<CPULightGroupSkySphere>;
extern template class RLBoundaryWork<CPULightGroupPoint>;
extern template class RLBoundaryWork<CPULightGroupDirectional>;
extern template class RLBoundaryWork<CPULightGroupSpot>;
extern template class RLBoundaryWork<CPULightGroupDisk>;
extern template class RLBoundaryWork<CPULightGroupRectangular>;
// ===================================================
// Path
extern template class RLWork<LambertCMat, GPUPrimitiveTriangle>;
extern template class RLWork<LambertCMat, GPUPrimitiveSphere>;

extern template class RLWork<ReflectMat, GPUPrimitiveTriangle>;
extern template class RLWork<ReflectMat, GPUPrimitiveSphere>;

extern template class RLWork<RefractMat, GPUPrimitiveTriangle>;
extern template class RLWork<RefractMat, GPUPrimitiveSphere>;

extern template class RLWork<LambertMat, GPUPrimitiveTriangle>;
extern template class RLWork<LambertMat, GPUPrimitiveSphere>;

extern template class RLWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class RLWork<UnrealMat, GPUPrimitiveSphere>;
// ===================================================
using RLBoundaryWorkerList = TypeList<RLBoundaryWork<CPULightGroupNull>,
                                      RLBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>,
                                      RLBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>,
                                      RLBoundaryWork<CPULightGroupSkySphere>,
                                      RLBoundaryWork<CPULightGroupPoint>,
                                      RLBoundaryWork<CPULightGroupDirectional>,
                                      RLBoundaryWork<CPULightGroupSpot>,
                                      RLBoundaryWork<CPULightGroupDisk>,
                                      RLBoundaryWork<CPULightGroupRectangular>>;
// ===================================================
using RLPathWorkerList = TypeList<RLWork<LambertCMat, GPUPrimitiveTriangle>,
                                  RLWork<LambertCMat, GPUPrimitiveSphere>,
                                  RLWork<ReflectMat, GPUPrimitiveTriangle>,
                                  RLWork<ReflectMat, GPUPrimitiveSphere>,
                                  RLWork<RefractMat, GPUPrimitiveTriangle>,
                                  RLWork<RefractMat, GPUPrimitiveSphere>,
                                  RLWork<LambertMat, GPUPrimitiveTriangle>,
                                  RLWork<LambertMat, GPUPrimitiveSphere>,
                                  RLWork<UnrealMat, GPUPrimitiveTriangle>,
                                  RLWork<UnrealMat, GPUPrimitiveSphere>>;