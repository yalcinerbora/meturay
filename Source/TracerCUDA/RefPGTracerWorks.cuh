#pragma once

// Materials
#include "DebugMaterials.cuh"
#include "SimpleMaterials.cuh"
#include "UnrealMaterial.cuh"
#include "LambertMaterial.cuh"
#include "EmptyMaterial.cuh"
#include "BoundaryMaterials.cuh"
// Primitives
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"
#include "GPUPrimitiveEmpty.h"
// Misc
#include "RefPGTracerKC.cuh"
#include "WorkPool.h"
#include "GPUWork.cuh"
#include "RayAuxStruct.cuh"

template<class MGroup, class PGroup>
class RPGBoundaryWork
    : public GPUBoundaryWorkBatch<RPGTracerGlobalState,
                                  RPGTracerLocalState, RayAuxPath,
                                  MGroup, PGroup, RPGTracerBoundaryWork<MGroup>,
                                  PGroup::GetSurfaceFunction>
{
    private:
        bool                            neeOn;
        bool                            misOn;

        using Base = GPUBoundaryWorkBatch<RPGTracerGlobalState,
                                          RPGTracerLocalState, RayAuxPath,
                                          MGroup, PGroup, RPGTracerBoundaryWork<MGroup>,
                                          PGroup::GetSurfaceFunction>;

    protected:
    public:
        // Constrcutors & Destructor
                                        RPGBoundaryWork(const GPUMaterialGroupI& mg,
                                                       const GPUPrimitiveGroupI& pg,
                                                       const GPUTransformI* const* t,
                                                       bool neeOn, bool misOn,
                                                       bool emptyPrimitive);
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

template<class M, class P>
RPGBoundaryWork<M, P>::RPGBoundaryWork(const GPUMaterialGroupI& mg,
                                       const GPUPrimitiveGroupI& pg,
                                       const GPUTransformI* const* t,
                                       bool neeOn, bool misOn,
                                       bool emptyPrimitive)
    : Base(mg, pg, t)
    , neeOn(neeOn)
    , misOn(misOn)
{
    // Populate localData
    this->localData.emptyPrimitive = emptyPrimitive;
}

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
        // Add one extra MIS ray for allocation
        // MIS ray is extra NEE ray for multiple importance sampling
        uint8_t misRay = (neeOn && misOn) ? 1 : 0;

        return this->materialGroup.SampleStrategyCount() + misRay + neeRay;
    }
    // Material does not require any samples meaning it is boundary material
    // Do not allocate any rays for this kind of material
    return 0;
}

// Ref PG Tracer Work Batches
// ===================================================
// Boundary
extern template class RPGBoundaryWork<BoundaryMatConstant, GPUPrimitiveEmpty>;
extern template class RPGBoundaryWork<BoundaryMatConstant, GPUPrimitiveTriangle>;
extern template class RPGBoundaryWork<BoundaryMatConstant, GPUPrimitiveSphere>;

extern template class RPGBoundaryWork<BoundaryMatTextured, GPUPrimitiveTriangle>;
extern template class RPGBoundaryWork<BoundaryMatTextured, GPUPrimitiveSphere>;

extern template class RPGBoundaryWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>;
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
using RPGBoundaryWorkerList = TypeList<RPGBoundaryWork<BoundaryMatConstant, GPUPrimitiveEmpty>,
                                       RPGBoundaryWork<BoundaryMatConstant, GPUPrimitiveTriangle>,
                                       RPGBoundaryWork<BoundaryMatConstant, GPUPrimitiveSphere>,
                                       RPGBoundaryWork<BoundaryMatTextured, GPUPrimitiveTriangle>,
                                       RPGBoundaryWork<BoundaryMatTextured, GPUPrimitiveSphere>,
                                       RPGBoundaryWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>>;
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