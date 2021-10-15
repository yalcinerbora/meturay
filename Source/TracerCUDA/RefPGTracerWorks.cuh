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
#include "RefPGTracerKC.cuh"
#include "WorkPool.h"
#include "GPUWork.cuh"
#include "RayAuxStruct.cuh"

template<class EGroup>
class RPGBoundaryWork final
    : public GPUBoundaryWorkBatch<RPGTracerGlobalState,
                                  RPGTracerLocalState, RayAuxPath,
                                  EGroup, RPGTracerBoundaryWork<EGroup>>
{
    private:
        bool                            neeOn;
        bool                            misOn;

        using Base = GPUBoundaryWorkBatch<RPGTracerGlobalState,
                                          RPGTracerLocalState, RayAuxPath,
                                          EGroup, RPGTracerBoundaryWork<EGroup>>;

    protected:
    public:
        // Constrcutors & Destructor
                                        RPGBoundaryWork(const CPUEndpointGroupI& eg,
                                                        const GPUTransformI* const* t,
                                                        bool neeOn, bool misOn);
                                        ~RPGBoundaryWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override { return 0; }

        const char*                     Type() const override { return Base::TypeName(); }
};


template<class MGroup, class PGroup>
class RPGPathWork
    : public GPUWorkBatch<RPGTracerGlobalState,
                          RPGTracerLocalState, RayAuxPath,
                          MGroup, PGroup, RPGTracerPathWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{

    private:
        bool                            neeOn;
        bool                            misOn;

        using Base = GPUWorkBatch<RPGTracerGlobalState,
                                  RPGTracerLocalState, RayAuxPath,
                                  MGroup, PGroup, RPGTracerPathWork<MGroup>,
                                  PGroup::GetSurfaceFunction>;
    protected:
    public:
        // Constrcutors & Destructor
                                        RPGPathWork(const GPUMaterialGroupI& mg,
                                                    const GPUPrimitiveGroupI& pg,
                                                    const GPUTransformI* const* t,
                                                    bool neeOn, bool misOn);
                                        ~RPGPathWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override;

        const char*                     Type() const override { return Base::TypeName(); }
};

template<class E>
RPGBoundaryWork<E>::RPGBoundaryWork(const CPUEndpointGroupI& eg,
                                    const GPUTransformI* const* t,
                                    bool neeOn, bool misOn)
    : Base(eg, t)
    , neeOn(neeOn)
    , misOn(misOn)
{}

template<class M, class P>
RPGPathWork<M, P>::RPGPathWork(const GPUMaterialGroupI& mg,
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
uint8_t RPGPathWork<M, P>::OutRayCount() const
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

// Ref PG Tracer Work Batches
// ===================================================
// Boundary
extern template class RPGBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>;
extern template class RPGBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>;
extern template class RPGBoundaryWork<CPULightGroupSkySphere>;
extern template class RPGBoundaryWork<CPULightGroupPoint>;
extern template class RPGBoundaryWork<CPULightGroupDirectional>;
extern template class RPGBoundaryWork<CPULightGroupSpot>;
extern template class RPGBoundaryWork<CPULightGroupDisk>;
extern template class RPGBoundaryWork<CPULightGroupRectangular>;
// ===================================================
// Path
extern template class RPGPathWork<LambertCMat, GPUPrimitiveTriangle>;
extern template class RPGPathWork<LambertCMat, GPUPrimitiveSphere>;

extern template class RPGPathWork<ReflectMat, GPUPrimitiveTriangle>;
extern template class RPGPathWork<ReflectMat, GPUPrimitiveSphere>;

extern template class RPGPathWork<RefractMat, GPUPrimitiveTriangle>;
extern template class RPGPathWork<RefractMat, GPUPrimitiveSphere>;

extern template class RPGPathWork<LambertMat, GPUPrimitiveTriangle>;
extern template class RPGPathWork<LambertMat, GPUPrimitiveSphere>;

extern template class RPGPathWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class RPGPathWork<UnrealMat, GPUPrimitiveSphere>;
// ===================================================
using RPGBoundaryWorkerList = TypeList<RPGBoundaryWork<CPULightGroupNull>,
                                       RPGBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>,
                                       RPGBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>,
                                       RPGBoundaryWork<CPULightGroupSkySphere>,
                                       RPGBoundaryWork<CPULightGroupPoint>,
                                       RPGBoundaryWork<CPULightGroupDirectional>,
                                       RPGBoundaryWork<CPULightGroupSpot>,
                                       RPGBoundaryWork<CPULightGroupDisk>,
                                       RPGBoundaryWork<CPULightGroupRectangular>>;
// ===================================================
using RPGPathWorkerList = TypeList<RPGPathWork<LambertCMat, GPUPrimitiveTriangle>,
                                   RPGPathWork<LambertCMat, GPUPrimitiveSphere>,
                                   RPGPathWork<ReflectMat, GPUPrimitiveTriangle>,
                                   RPGPathWork<ReflectMat, GPUPrimitiveSphere>,
                                   RPGPathWork<RefractMat, GPUPrimitiveTriangle>,
                                   RPGPathWork<RefractMat, GPUPrimitiveSphere>,
                                   RPGPathWork<LambertMat, GPUPrimitiveTriangle>,
                                   RPGPathWork<LambertMat, GPUPrimitiveSphere>,
                                   RPGPathWork<UnrealMat, GPUPrimitiveTriangle>,
                                   RPGPathWork<UnrealMat, GPUPrimitiveSphere>>;