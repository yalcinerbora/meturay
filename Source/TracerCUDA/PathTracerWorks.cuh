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
    : public GPUWorkBatch<PathTracerGlobalState, 
                          PathTracerLocalState, RayAuxPath,
                          MGroup, PGroup, PathTracerBoundaryWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
        bool                            neeOn;
        bool                            misOn;

    protected:
    public:
        // Constrcutors & Destructor
                                        PTBoundaryWork(const GPUMaterialGroupI& mg,
                                                               const GPUPrimitiveGroupI& pg,
                                                               const GPUTransformI* const* t,
                                                               bool neeOn, bool misOn,
                                                               bool emptyPrimitive);
                                        ~PTBoundaryWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override { return 0; }

        const char*                     Type() const override { return TypeName(); }
};

template<class MGroup, class PGroup>
class PTComboWork
    : public GPUWorkBatch<PathTracerGlobalState, 
                          PathTracerLocalState, RayAuxPath,
                          MGroup, PGroup, PathTracerComboWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
        bool                            neeOn;
        bool                            misOn;

    protected:
    public:
        // Constrcutors & Destructor
                                        PTComboWork(const GPUMaterialGroupI& mg,
                                                   const GPUPrimitiveGroupI& pg,
                                                   const GPUTransformI* const* t,
                                                    bool neeOn, bool misOn);
                                        ~PTComboWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override;

        const char*                     Type() const override { return TypeName(); }
};

template<class MGroup, class PGroup>
class PTPathWork
    : public GPUWorkBatch<PathTracerGlobalState, 
                          PathTracerLocalState, RayAuxPath,
                          MGroup, PGroup, PathTracerPathWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
    protected:
    public:
        // Constrcutors & Destructor
                                        PTPathWork(const GPUMaterialGroupI& mg,
                                                   const GPUPrimitiveGroupI& pg,
                                                   const GPUTransformI* const* t);
                                        ~PTPathWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override;

        const char*                     Type() const override { return TypeName(); }
};

template<class MGroup, class PGroup>
class PTNEEWork
    : public GPUWorkBatch<PathTracerGlobalState, 
                          PathTracerLocalState, RayAuxPath,
                          MGroup, PGroup, PathTracerPathWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
        bool                            misOn;

    protected:
    public:
        // Constrcutors & Destructor
                                        PTNEEWork(const GPUMaterialGroupI& mg,
                                                   const GPUPrimitiveGroupI& pg,
                                                   const GPUTransformI* const* t,
                                                   bool misOn);
                                        ~PTNEEWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override;

        const char*                     Type() const override { return TypeName(); }
};

template<class MGroup, class PGroup>
class PTMISWork
    : public GPUWorkBatch<PathTracerGlobalState, 
                          PathTracerLocalState, RayAuxPath,
                          MGroup, PGroup, PathTracerPathWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
    protected:
    public:
        // Constrcutors & Destructor
                                        PTMISWork(const GPUMaterialGroupI& mg,
                                                   const GPUPrimitiveGroupI& pg,
                                                   const GPUTransformI* const* t);
                                        ~PTMISWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override;

        const char*                     Type() const override { return TypeName(); }
};

template<class M, class P>
PTBoundaryWork<M, P>::PTBoundaryWork(const GPUMaterialGroupI& mg,
                                     const GPUPrimitiveGroupI& pg,
                                     const GPUTransformI* const* t,
                                     bool neeOn, bool misOn,
                                     bool emptyPrimitive)
    : GPUWorkBatch<PathTracerGlobalState, 
                   PathTracerLocalState, RayAuxPath,
                   M, P, PathTracerBoundaryWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
    , neeOn(neeOn)
    , misOn(misOn)
{
    // Populate localData
    localData.emptyPrimitive = emptyPrimitive;
}

template<class M, class P>
PTComboWork<M, P>::PTComboWork(const GPUMaterialGroupI& mg,
                               const GPUPrimitiveGroupI& pg,
                               const GPUTransformI* const* t,
                               bool neeOn, bool misOn)
    : GPUWorkBatch<PathTracerGlobalState, 
                   PathTracerLocalState, RayAuxPath,
                   M, P, PathTracerPathWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
    , neeOn(neeOn)
    , misOn(misOn)
{
    // Populate localData
    localData.emptyPrimitive = false;
}

template<class M, class P>
uint8_t PTComboWork<M, P>::OutRayCount() const
{
    // Incorporate NEE Ray as an addition
    // If material can be sampled (i.e no Dirac Delta BxDF)
    if(!materialGroup.CanBeSampled())
    {
        // Material Cannot be sampled just allocate whatever
        // the material is requesting
        return materialGroup.SampleStrategyCount();
    }
    else if(materialGroup.SampleStrategyCount() != 0)
    {
        // Material can be sampled 
        // Add one extra nee ray allocation
        uint8_t neeRay = (neeOn) ? 1 : 0;
        // Add one extra MIS ray for allocation
        // MIS ray is extra NEE ray for multiple importance sampling
        uint8_t misRay = (neeOn && misOn) ? 1 : 0;

        return materialGroup.SampleStrategyCount() + misRay + neeRay;
    }
    // Material does not require any samples meaning it is boundary material
    // Do not allocate any rays for this kind of material
    return 0;
}

template<class M, class P>
PTPathWork<M, P>::PTPathWork(const GPUMaterialGroupI& mg,
                             const GPUPrimitiveGroupI& pg,
                             const GPUTransformI* const* t)
    : GPUWorkBatch<PathTracerGlobalState, 
                   PathTracerLocalState, RayAuxPath,
                   M, P, PathTracerPathWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
{
    // Populate localData
    localData.emptyPrimitive = false;   
}

template<class M, class P>
uint8_t PTPathWork<M, P>::OutRayCount() const
{
    return materialGroup.SampleStrategyCount();
}

template<class M, class P>
PTNEEWork<M, P>::PTNEEWork(const GPUMaterialGroupI& mg,
                             const GPUPrimitiveGroupI& pg,
                             const GPUTransformI* const* t,
                             bool misOn)
    : GPUWorkBatch<PathTracerGlobalState, 
                   PathTracerLocalState, RayAuxPath,
                   M, P, PathTracerNEEWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
    , misOn(misOn)
{
    // Populate localData
    localData.emptyPrimitive = false;   
}

template<class M, class P>
uint8_t PTNEEWork<M, P>::OutRayCount() const
{
    // If material can be sampled
    // non-zero radiance NEE Ray can be generated
    // else dont generate ray
    uint8_t neeRay = (materialGroup.CanBeSampled()) ? 1 : 0;
    return neeRay;
}

template<class M, class P>
PTMISWork<M, P>::PTMISWork(const GPUMaterialGroupI& mg,
                           const GPUPrimitiveGroupI& pg,
                           const GPUTransformI* const* t)
    : GPUWorkBatch<PathTracerGlobalState, 
                   PathTracerLocalState, RayAuxPath,
                   M, P, PathTracerMISWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
{
    // Populate localData
    localData.emptyPrimitive = false;   
}

template<class M, class P>
uint8_t PTMISWork<M, P>::OutRayCount() const
{
    // If material can be sampled
    // non-zero radiance NEE Ray can be generated
    // else dont generate ray
    uint8_t misRay = (materialGroup.CanBeSampled()) ? 1 : 0;
    return misRay;
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
// Combo
extern template class PTComboWork<LambertCMat, GPUPrimitiveTriangle>;
extern template class PTComboWork<LambertCMat, GPUPrimitiveSphere>;

extern template class PTComboWork<ReflectMat, GPUPrimitiveTriangle>;
extern template class PTComboWork<ReflectMat, GPUPrimitiveSphere>;

extern template class PTComboWork<RefractMat, GPUPrimitiveTriangle>;
extern template class PTComboWork<RefractMat, GPUPrimitiveSphere>;

extern template class PTComboWork<LambertMat, GPUPrimitiveTriangle>;
extern template class PTComboWork<LambertMat, GPUPrimitiveSphere>;

extern template class PTComboWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class PTComboWork<UnrealMat, GPUPrimitiveSphere>;
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
// NEE
extern template class PTNEEWork<LambertCMat, GPUPrimitiveTriangle>;
extern template class PTNEEWork<LambertCMat, GPUPrimitiveSphere>;

extern template class PTNEEWork<ReflectMat, GPUPrimitiveTriangle>;
extern template class PTNEEWork<ReflectMat, GPUPrimitiveSphere>;

extern template class PTNEEWork<RefractMat, GPUPrimitiveTriangle>;
extern template class PTNEEWork<RefractMat, GPUPrimitiveSphere>;

extern template class PTNEEWork<LambertMat, GPUPrimitiveTriangle>;
extern template class PTNEEWork<LambertMat, GPUPrimitiveSphere>;

extern template class PTNEEWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class PTNEEWork<UnrealMat, GPUPrimitiveSphere>;
// ===================================================
// MIS
extern template class PTMISWork<LambertCMat, GPUPrimitiveTriangle>;
extern template class PTMISWork<LambertCMat, GPUPrimitiveSphere>;

extern template class PTMISWork<ReflectMat, GPUPrimitiveTriangle>;
extern template class PTMISWork<ReflectMat, GPUPrimitiveSphere>;

extern template class PTMISWork<RefractMat, GPUPrimitiveTriangle>;
extern template class PTMISWork<RefractMat, GPUPrimitiveSphere>;

extern template class PTMISWork<LambertMat, GPUPrimitiveTriangle>;
extern template class PTMISWork<LambertMat, GPUPrimitiveSphere>;

extern template class PTMISWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class PTMISWork<UnrealMat, GPUPrimitiveSphere>;
// ===================================================
using PTBoundaryWorkerList = TypeList<PTBoundaryWork<BoundaryMatConstant, GPUPrimitiveEmpty>,
                                      PTBoundaryWork<BoundaryMatConstant, GPUPrimitiveTriangle>,
                                      PTBoundaryWork<BoundaryMatConstant, GPUPrimitiveSphere>,
                                      PTBoundaryWork<BoundaryMatTextured, GPUPrimitiveTriangle>,
                                      PTBoundaryWork<BoundaryMatTextured, GPUPrimitiveSphere>,
                                      PTBoundaryWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>>;
// ===================================================
using PTComboWorkerList = TypeList<PTComboWork<LambertCMat, GPUPrimitiveTriangle>,
                                   PTComboWork<LambertCMat, GPUPrimitiveSphere>,
                                   PTComboWork<ReflectMat, GPUPrimitiveTriangle>,
                                   PTComboWork<ReflectMat, GPUPrimitiveSphere>,
                                   PTComboWork<RefractMat, GPUPrimitiveTriangle>,
                                   PTComboWork<RefractMat, GPUPrimitiveSphere>,
                                   PTComboWork<LambertMat, GPUPrimitiveTriangle>,
                                   PTComboWork<LambertMat, GPUPrimitiveSphere>,
                                   PTComboWork<UnrealMat, GPUPrimitiveTriangle>,
                                   PTComboWork<UnrealMat, GPUPrimitiveSphere>>;
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
// ===================================================
using PTNEEWorkerList = TypeList<PTNEEWork<LambertCMat, GPUPrimitiveTriangle>,
                                 PTNEEWork<LambertCMat, GPUPrimitiveSphere>,
                                 PTNEEWork<ReflectMat, GPUPrimitiveTriangle>,
                                 PTNEEWork<ReflectMat, GPUPrimitiveSphere>,
                                 PTNEEWork<RefractMat, GPUPrimitiveTriangle>,
                                 PTNEEWork<RefractMat, GPUPrimitiveSphere>,
                                 PTNEEWork<LambertMat, GPUPrimitiveTriangle>,
                                 PTNEEWork<LambertMat, GPUPrimitiveSphere>,
                                 PTNEEWork<UnrealMat, GPUPrimitiveTriangle>,
                                 PTNEEWork<UnrealMat, GPUPrimitiveSphere>>;
// ===================================================
using PTMISWorkerList = TypeList<PTMISWork<LambertCMat, GPUPrimitiveTriangle>,
                                 PTMISWork<LambertCMat, GPUPrimitiveSphere>,
                                 PTMISWork<ReflectMat, GPUPrimitiveTriangle>,
                                 PTMISWork<ReflectMat, GPUPrimitiveSphere>,
                                 PTMISWork<RefractMat, GPUPrimitiveTriangle>,
                                 PTMISWork<RefractMat, GPUPrimitiveSphere>,
                                 PTMISWork<LambertMat, GPUPrimitiveTriangle>,
                                 PTMISWork<LambertMat, GPUPrimitiveSphere>,
                                 PTMISWork<UnrealMat, GPUPrimitiveTriangle>,
                                 PTMISWork<UnrealMat, GPUPrimitiveSphere>>;