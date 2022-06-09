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
#include "GPULightConstant.cuh"
// Misc
#include "WFPGTracerKC.cuh"
#include "WorkPool.h"
#include "GPUWork.cuh"
#include "RayAuxStruct.cuh"

template<class EGroup>
class WFPGBoundaryWork
    : public GPUBoundaryWorkBatch<WFPGTracerGlobalState,
                                  WFPGTracerLocalState, RayAuxWFPG,
                                  EGroup, WFPGTracerBoundaryWork<EGroup>>
{
    private:
        bool                            neeOn;
        bool                            misOn;

        using Base = GPUBoundaryWorkBatch<WFPGTracerGlobalState,
                                          WFPGTracerLocalState, RayAuxWFPG,
                                          EGroup, WFPGTracerBoundaryWork<EGroup>>;

    protected:
    public:
        // Constructors & Destructor
                                        WFPGBoundaryWork(const CPUEndpointGroupI& eg,
                                                         const GPUTransformI* const* t,
                                                         bool neeOn, bool misOn);
                                        ~WFPGBoundaryWork() = default;

        void                            GetReady() override {}
        const char*                     Type() const override { return Base::TypeName(); }
};

template<class MGroup, class PGroup>
class WFPGWork
    : public GPUWorkBatch<WFPGTracerGlobalState,
                          WFPGTracerLocalState, RayAuxWFPG,
                          MGroup, PGroup, WFPGTracerPathWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
        bool                            neeOn;
        bool                            misOn;

        using Base = GPUWorkBatch<WFPGTracerGlobalState,
                                  WFPGTracerLocalState, RayAuxWFPG,
                                  MGroup, PGroup, WFPGTracerPathWork<MGroup>,
                                  PGroup::GetSurfaceFunction>;

    protected:
    public:
        // Constructors & Destructor
                        WFPGWork(const GPUMaterialGroupI& mg,
                               const GPUPrimitiveGroupI& pg,
                               const GPUTransformI* const* t,
                               bool neeOn, bool misOn);
                        ~WFPGWork() = default;

        void            GetReady() override {}
        uint8_t         OutRayCount() const override;
        const char*     Type() const override { return Base::TypeName(); }
};


template<class MGroup, class PGroup>
class WFPGPhotonWork
    : public GPUWorkBatch<WFPGTracerGlobalState,
                          WFPGTracerLocalState, RayAuxPhotonWFPG,
                          MGroup, PGroup, WFPGTracerPhotonWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
        bool                            neeOn;
        bool                            misOn;

        using Base = GPUWorkBatch<WFPGTracerGlobalState,
                                  WFPGTracerLocalState, RayAuxPhotonWFPG,
                                  MGroup, PGroup, WFPGTracerPhotonWork<MGroup>,
                                  PGroup::GetSurfaceFunction>;

    protected:
    public:
        // Constructors & Destructor
                        WFPGPhotonWork(const GPUMaterialGroupI& mg,
                                       const GPUPrimitiveGroupI& pg,
                                       const GPUTransformI* const* t);
                        ~WFPGPhotonWork() = default;

        void            GetReady() override {}
        uint8_t         OutRayCount() const override { return 1; };
        const char*     Type() const override { return Base::TypeName(); }
};

template<class EGroup>
class WFPGDebugBoundaryWork
    : public GPUBoundaryWorkBatch<WFPGTracerGlobalState,
                                  WFPGTracerLocalState, RayAuxWFPG,
                                  EGroup, WFPGTracerDebugBWork<EGroup>>
{
    private:
        using Base = GPUBoundaryWorkBatch<WFPGTracerGlobalState,
                                          WFPGTracerLocalState, RayAuxWFPG,
                                          EGroup, WFPGTracerDebugBWork<EGroup>>;

    public:
                            WFPGDebugBoundaryWork(const CPUEndpointGroupI& eg,
                                                const GPUTransformI* const* t);
                            ~WFPGDebugBoundaryWork() = default;

        void                GetReady() override {}
        const char*         Type() const override { return Base::TypeName(); }
};

template<class MGroup, class PGroup>
class WFPGDebugWork
    : public GPUWorkBatch<WFPGTracerGlobalState,
                          WFPGTracerLocalState, RayAuxWFPG,
                          MGroup, PGroup, WFPGTracerDebugWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
        using Base = GPUWorkBatch<WFPGTracerGlobalState,
                                  WFPGTracerLocalState, RayAuxWFPG,
                                  MGroup, PGroup, WFPGTracerDebugWork<MGroup>,
                                  PGroup::GetSurfaceFunction>;

    public:
                            WFPGDebugWork(const GPUMaterialGroupI& mg,
                                          const GPUPrimitiveGroupI& pg,
                                          const GPUTransformI* const* t);
                            ~WFPGDebugWork() = default;

        void                GetReady() override {}
        uint8_t             OutRayCount() const override;
        const char*         Type() const override { return Base::TypeName(); }
};

template<class E>
WFPGBoundaryWork<E>::WFPGBoundaryWork(const CPUEndpointGroupI& eg,
                                      const GPUTransformI* const* t,
                                      bool neeOn, bool misOn)
    : Base(eg, t)
    , neeOn(neeOn)
    , misOn(misOn)
{}

template<class M, class P>
WFPGWork<M, P>::WFPGWork(const GPUMaterialGroupI& mg,
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
WFPGPhotonWork<M, P>::WFPGPhotonWork(const GPUMaterialGroupI& mg,
                                     const GPUPrimitiveGroupI& pg,
                                     const GPUTransformI* const* t)
    : Base(mg, pg, t)
{
    // Populate localData
    this->localData.emptyPrimitive = false;
}

template<class E>
WFPGDebugBoundaryWork<E>::WFPGDebugBoundaryWork(const CPUEndpointGroupI& eg,
                                                const GPUTransformI* const* t)
    : Base(eg, t)
{}

template<class M, class P>
WFPGDebugWork<M, P>::WFPGDebugWork(const GPUMaterialGroupI& mg,
                                   const GPUPrimitiveGroupI& pg,
                                   const GPUTransformI* const* t)
    : Base(mg, pg, t)
{}

template<class M, class P>
uint8_t WFPGWork<M, P>::OutRayCount() const
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

template<class M, class P>
uint8_t WFPGDebugWork<M, P>::OutRayCount() const
{
    return 0;
}

// PPG Tracer Work Batches
// ===================================================
// Boundary
extern template class WFPGBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>;
extern template class WFPGBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>;
extern template class WFPGBoundaryWork<CPULightGroupSkySphere>;
extern template class WFPGBoundaryWork<CPULightGroupPoint>;
extern template class WFPGBoundaryWork<CPULightGroupDirectional>;
extern template class WFPGBoundaryWork<CPULightGroupSpot>;
extern template class WFPGBoundaryWork<CPULightGroupDisk>;
extern template class WFPGBoundaryWork<CPULightGroupRectangular>;
// Debug Boundary
extern template class WFPGDebugBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>;
extern template class WFPGDebugBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>;
extern template class WFPGDebugBoundaryWork<CPULightGroupSkySphere>;
extern template class WFPGDebugBoundaryWork<CPULightGroupPoint>;
extern template class WFPGDebugBoundaryWork<CPULightGroupDirectional>;
extern template class WFPGDebugBoundaryWork<CPULightGroupSpot>;
extern template class WFPGDebugBoundaryWork<CPULightGroupDisk>;
extern template class WFPGDebugBoundaryWork<CPULightGroupRectangular>;
extern template class WFPGDebugBoundaryWork<CPULightGroupConstant>;
// ===================================================
// Path
extern template class WFPGWork<LambertCMat, GPUPrimitiveTriangle>;
extern template class WFPGWork<LambertCMat, GPUPrimitiveSphere>;

extern template class WFPGWork<ReflectMat, GPUPrimitiveTriangle>;
extern template class WFPGWork<ReflectMat, GPUPrimitiveSphere>;

extern template class WFPGWork<RefractMat, GPUPrimitiveTriangle>;
extern template class WFPGWork<RefractMat, GPUPrimitiveSphere>;

extern template class WFPGWork<LambertMat, GPUPrimitiveTriangle>;
extern template class WFPGWork<LambertMat, GPUPrimitiveSphere>;

extern template class WFPGWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class WFPGWork<UnrealMat, GPUPrimitiveSphere>;
// Photon Path
extern template class WFPGPhotonWork<LambertCMat, GPUPrimitiveTriangle>;
extern template class WFPGPhotonWork<LambertCMat, GPUPrimitiveSphere>;

extern template class WFPGPhotonWork<ReflectMat, GPUPrimitiveTriangle>;
extern template class WFPGPhotonWork<ReflectMat, GPUPrimitiveSphere>;

extern template class WFPGPhotonWork<RefractMat, GPUPrimitiveTriangle>;
extern template class WFPGPhotonWork<RefractMat, GPUPrimitiveSphere>;

extern template class WFPGPhotonWork<LambertMat, GPUPrimitiveTriangle>;
extern template class WFPGPhotonWork<LambertMat, GPUPrimitiveSphere>;

extern template class WFPGPhotonWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class WFPGPhotonWork<UnrealMat, GPUPrimitiveSphere>;
// Debug Path
extern template class WFPGDebugWork<LambertCMat, GPUPrimitiveTriangle>;
extern template class WFPGDebugWork<LambertCMat, GPUPrimitiveSphere>;

extern template class WFPGDebugWork<ReflectMat, GPUPrimitiveTriangle>;
extern template class WFPGDebugWork<ReflectMat, GPUPrimitiveSphere>;

extern template class WFPGDebugWork<RefractMat, GPUPrimitiveTriangle>;
extern template class WFPGDebugWork<RefractMat, GPUPrimitiveSphere>;

extern template class WFPGDebugWork<LambertMat, GPUPrimitiveTriangle>;
extern template class WFPGDebugWork<LambertMat, GPUPrimitiveSphere>;

extern template class WFPGDebugWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class WFPGDebugWork<UnrealMat, GPUPrimitiveSphere>;
// ===================================================
using WFPGBoundaryWorkerList = TypeList<WFPGBoundaryWork<CPULightGroupNull>,
                                        WFPGBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>,
                                        WFPGBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>,
                                        WFPGBoundaryWork<CPULightGroupSkySphere>,
                                        WFPGBoundaryWork<CPULightGroupPoint>,
                                        WFPGBoundaryWork<CPULightGroupDirectional>,
                                        WFPGBoundaryWork<CPULightGroupSpot>,
                                        WFPGBoundaryWork<CPULightGroupDisk>,
                                        WFPGBoundaryWork<CPULightGroupRectangular>>;
// ===================================================
using WFPGDebugBoundaryWorkerList = TypeList<WFPGDebugBoundaryWork<CPULightGroupNull>,
                                             WFPGDebugBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>,
                                             WFPGDebugBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>,
                                             WFPGDebugBoundaryWork<CPULightGroupSkySphere>,
                                             WFPGDebugBoundaryWork<CPULightGroupPoint>,
                                             WFPGDebugBoundaryWork<CPULightGroupDirectional>,
                                             WFPGDebugBoundaryWork<CPULightGroupSpot>,
                                             WFPGDebugBoundaryWork<CPULightGroupDisk>,
                                             WFPGDebugBoundaryWork<CPULightGroupRectangular>,
                                             WFPGDebugBoundaryWork<CPULightGroupConstant>>;
// ===================================================
using WFPGPathWorkerList = TypeList<WFPGWork<LambertCMat, GPUPrimitiveTriangle>,
                                    WFPGWork<LambertCMat, GPUPrimitiveSphere>,
                                    WFPGWork<ReflectMat, GPUPrimitiveTriangle>,
                                    WFPGWork<ReflectMat, GPUPrimitiveSphere>,
                                    WFPGWork<RefractMat, GPUPrimitiveTriangle>,
                                    WFPGWork<RefractMat, GPUPrimitiveSphere>,
                                    WFPGWork<LambertMat, GPUPrimitiveTriangle>,
                                    WFPGWork<LambertMat, GPUPrimitiveSphere>,
                                    WFPGWork<UnrealMat, GPUPrimitiveTriangle>,
                                    WFPGWork<UnrealMat, GPUPrimitiveSphere>>;
// ===================================================
using WFPGPhotonWorkerList = TypeList<WFPGPhotonWork<LambertCMat, GPUPrimitiveTriangle>,
                                      WFPGPhotonWork<LambertCMat, GPUPrimitiveSphere>,
                                      WFPGPhotonWork<ReflectMat, GPUPrimitiveTriangle>,
                                      WFPGPhotonWork<ReflectMat, GPUPrimitiveSphere>,
                                      WFPGPhotonWork<RefractMat, GPUPrimitiveTriangle>,
                                      WFPGPhotonWork<RefractMat, GPUPrimitiveSphere>,
                                      WFPGPhotonWork<LambertMat, GPUPrimitiveTriangle>,
                                      WFPGPhotonWork<LambertMat, GPUPrimitiveSphere>,
                                      WFPGPhotonWork<UnrealMat, GPUPrimitiveTriangle>,
                                      WFPGPhotonWork<UnrealMat, GPUPrimitiveSphere>>;
// ===================================================
using WFPGDebugPathWorkerList = TypeList<WFPGDebugWork<LambertCMat, GPUPrimitiveTriangle>,
                                         WFPGDebugWork<LambertCMat, GPUPrimitiveSphere>,
                                         WFPGDebugWork<ReflectMat, GPUPrimitiveTriangle>,
                                         WFPGDebugWork<ReflectMat, GPUPrimitiveSphere>,
                                         WFPGDebugWork<RefractMat, GPUPrimitiveTriangle>,
                                         WFPGDebugWork<RefractMat, GPUPrimitiveSphere>,
                                         WFPGDebugWork<LambertMat, GPUPrimitiveTriangle>,
                                         WFPGDebugWork<LambertMat, GPUPrimitiveSphere>,
                                         WFPGDebugWork<UnrealMat, GPUPrimitiveTriangle>,
                                         WFPGDebugWork<UnrealMat, GPUPrimitiveSphere>>;