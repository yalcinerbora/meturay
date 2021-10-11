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
#include "PathTracerKC.cuh"
#include "WorkPool.h"
#include "GPUWork.cuh"
#include "RayAuxStruct.cuh"

template<class MGroup, class PGroup>
class PTBoundaryWork
    : public GPUBoundaryWorkBatch<PathTracerGlobalState,
                                  PathTracerLocalState, RayAuxPath,
                                  MGroup, PGroup, PathTracerBoundaryWork<MGroup>,
                                  PGroup::GetSurfaceFunction>
{
    private:
        bool                neeOn;
        bool                misOn;

        using Base = GPUBoundaryWorkBatch<PathTracerGlobalState,
                                          PathTracerLocalState, RayAuxPath,
                                          MGroup, PGroup, PathTracerBoundaryWork<MGroup>,
                                          PGroup::GetSurfaceFunction>;

    protected:
    public:
        // Constrcutors & Destructor
                                PTBoundaryWork(const GPUMaterialGroupI& mg,
                                               const GPUPrimitiveGroupI& pg,
                                               const GPUTransformI* const* t,
                                               bool neeOn, bool misOn,
                                               bool emptyPrimitive);
                                ~PTBoundaryWork() = default;

        void                    GetReady() override {}
        uint8_t                 OutRayCount() const override { return 0; }

        const char*             Type() const override { return Base::TypeName(); }
};

template<class MGroup, class PGroup>
class PTPathWork
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
        // Constrcutors & Destructor
                            PTPathWork(const GPUMaterialGroupI& mg,
                                        const GPUPrimitiveGroupI& pg,
                                        const GPUTransformI* const* t,
                                        bool neeOn, bool misOn);
                            ~PTPathWork() = default;

        void                GetReady() override {}
        uint8_t             OutRayCount() const override;

        const char*         Type() const override { return Base::TypeName(); }
};

template<class M, class P>
PTBoundaryWork<M, P>::PTBoundaryWork(const GPUMaterialGroupI& mg,
                                     const GPUPrimitiveGroupI& pg,
                                     const GPUTransformI* const* t,
                                     bool neeOn, bool misOn,
                                     bool emptyPrimitive)
    : Base(mg, pg, t)
    , neeOn(neeOn)
    , misOn(neeOn && misOn)
{
    // Populate localData
    this->localData.emptyPrimitive = emptyPrimitive;
}

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
extern template class PTBoundaryWork<BoundaryMatConstant, GPUPrimitiveEmpty>;
extern template class PTBoundaryWork<BoundaryMatConstant, GPUPrimitiveTriangle>;
extern template class PTBoundaryWork<BoundaryMatConstant, GPUPrimitiveSphere>;

extern template class PTBoundaryWork<BoundaryMatTextured, GPUPrimitiveTriangle>;
extern template class PTBoundaryWork<BoundaryMatTextured, GPUPrimitiveSphere>;

extern template class PTBoundaryWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>;
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
using PTBoundaryWorkerList = TypeList<PTBoundaryWork<BoundaryMatConstant, GPUPrimitiveEmpty>,
                                      PTBoundaryWork<BoundaryMatConstant, GPUPrimitiveTriangle>,
                                      PTBoundaryWork<BoundaryMatConstant, GPUPrimitiveSphere>,
                                      PTBoundaryWork<BoundaryMatTextured, GPUPrimitiveTriangle>,
                                      PTBoundaryWork<BoundaryMatTextured, GPUPrimitiveSphere>,
                                      PTBoundaryWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>>;
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