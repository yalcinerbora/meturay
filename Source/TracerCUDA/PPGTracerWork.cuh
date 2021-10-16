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
#include "PPGTracerKC.cuh"
#include "WorkPool.h"
#include "GPUWork.cuh"
#include "RayAuxStruct.cuh"

template<class EGroup>
class PPGBoundaryWork
    : public GPUBoundaryWorkBatch<PPGTracerGlobalState,
                                  PPGTracerLocalState, RayAuxPPG,
                                  EGroup, PPGTracerBoundaryWork<EGroup>>
{
    private:
        bool                            neeOn;
        bool                            misOn;

        using Base = GPUBoundaryWorkBatch<PPGTracerGlobalState,
                                          PPGTracerLocalState, RayAuxPPG,
                                          EGroup, PPGTracerBoundaryWork<EGroup>>;

    protected:
    public:
        // Constrcutors & Destructor
                                        PPGBoundaryWork(const CPUEndpointGroupI& eg,
                                                        const GPUTransformI* const* t,
                                                        bool neeOn, bool misOn);
                                        ~PPGBoundaryWork() = default;

        void                            GetReady() override {}

        const char*                     Type() const override { return Base::TypeName(); }
};

template<class MGroup, class PGroup>
class PPGWork
    : public GPUWorkBatch<PPGTracerGlobalState,
                          PPGTracerLocalState, RayAuxPPG,
                          MGroup, PGroup, PPGTracerPathWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
        bool                            neeOn;
        bool                            misOn;

        using Base = GPUWorkBatch<PPGTracerGlobalState,
                                  PPGTracerLocalState, RayAuxPPG,
                                  MGroup, PGroup, PPGTracerPathWork<MGroup>,
                                  PGroup::GetSurfaceFunction>;

    protected:
    public:
        // Constrcutors & Destructor
                                        PPGWork(const GPUMaterialGroupI& mg,
                                                const GPUPrimitiveGroupI& pg,
                                                const GPUTransformI* const* t,
                                                bool neeOn, bool misOn);
                                        ~PPGWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override;

        const char*                     Type() const override { return Base::TypeName(); }
};

template<class E>
PPGBoundaryWork<E>::PPGBoundaryWork(const CPUEndpointGroupI& eg,
                                    const GPUTransformI* const* t,
                                    bool neeOn, bool misOn)
    : Base(eg, t)
    , neeOn(neeOn)
    , misOn(misOn)
{}

template<class M, class P>
PPGWork<M, P>::PPGWork(const GPUMaterialGroupI& mg,
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
uint8_t PPGWork<M, P>::OutRayCount() const
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
extern template class PPGBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>;
extern template class PPGBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>;
extern template class PPGBoundaryWork<CPULightGroupSkySphere>;
extern template class PPGBoundaryWork<CPULightGroupPoint>;
extern template class PPGBoundaryWork<CPULightGroupDirectional>;
extern template class PPGBoundaryWork<CPULightGroupSpot>;
extern template class PPGBoundaryWork<CPULightGroupDisk>;
extern template class PPGBoundaryWork<CPULightGroupRectangular>;
// ===================================================
// Path
extern template class PPGWork<LambertCMat, GPUPrimitiveTriangle>;
extern template class PPGWork<LambertCMat, GPUPrimitiveSphere>;

extern template class PPGWork<ReflectMat, GPUPrimitiveTriangle>;
extern template class PPGWork<ReflectMat, GPUPrimitiveSphere>;

extern template class PPGWork<RefractMat, GPUPrimitiveTriangle>;
extern template class PPGWork<RefractMat, GPUPrimitiveSphere>;

extern template class PPGWork<LambertMat, GPUPrimitiveTriangle>;
extern template class PPGWork<LambertMat, GPUPrimitiveSphere>;

extern template class PPGWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class PPGWork<UnrealMat, GPUPrimitiveSphere>;
// ===================================================
using PPGBoundaryWorkerList = TypeList<PPGBoundaryWork<CPULightGroupNull>,
                                       PPGBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>,
                                       PPGBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>,
                                       PPGBoundaryWork<CPULightGroupSkySphere>,
                                       PPGBoundaryWork<CPULightGroupPoint>,
                                       PPGBoundaryWork<CPULightGroupDirectional>,
                                       PPGBoundaryWork<CPULightGroupSpot>,
                                       PPGBoundaryWork<CPULightGroupDisk>,
                                       PPGBoundaryWork<CPULightGroupRectangular>>;
// ===================================================
using PPGPathWorkerList = TypeList<PPGWork<LambertCMat, GPUPrimitiveTriangle>,
                                   PPGWork<LambertCMat, GPUPrimitiveSphere>,
                                   PPGWork<ReflectMat, GPUPrimitiveTriangle>,
                                   PPGWork<ReflectMat, GPUPrimitiveSphere>,
                                   PPGWork<RefractMat, GPUPrimitiveTriangle>,
                                   PPGWork<RefractMat, GPUPrimitiveSphere>,
                                   PPGWork<LambertMat, GPUPrimitiveTriangle>,
                                   PPGWork<LambertMat, GPUPrimitiveSphere>,
                                   PPGWork<UnrealMat, GPUPrimitiveTriangle>,
                                   PPGWork<UnrealMat, GPUPrimitiveSphere>>;