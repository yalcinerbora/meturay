#pragma once

#include "DebugMaterials.cuh"
#include "SimpleMaterials.cuh"
#include "UnrealMaterial.cuh"
#include "LambertMaterial.cuh"
#include "EmptyMaterial.cuh"
#include "TracerKC.cuh"

#include "WorkPool.h"
#include "GPUWork.cuh"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"
#include "GPUPrimitiveEmpty.h"
#include "BoundaryMaterials.cuh"

template<class MGroup, class PGroup>
class DirectTracerWork
    : public GPUWorkBatch<DirectTracerGlobal, EmptyState, RayAuxBasic,
                          MGroup, PGroup, BasicWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
    protected:
    public:
        // Constrcutors & Destructor
                                        DirectTracerWork(const GPUMaterialGroupI& mg,
                                                         const GPUPrimitiveGroupI& pg,
                                                         const GPUTransformI* const* t);
                                        ~DirectTracerWork() = default;

        void                            GetReady() override {}
        // We will not bounce more than once
        uint8_t                         OutRayCount() const override { return 0; }

        const char*                     Type() const override { return TypeName(); }
};

template<class MGroup, class PGroup>
class PathTracerWork
    : public GPUWorkBatch<PathTracerGlobal, PathTracerLocal, RayAuxPath,
                          MGroup, PGroup, PathWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
        bool                            neeOn;
        bool                            misOn;

    protected:
    public:
        // Constrcutors & Destructor
                                        PathTracerWork(const GPUMaterialGroupI& mg,
                                                       const GPUPrimitiveGroupI& pg,
                                                       const GPUTransformI* const* t,
                                                       bool neeOn, bool misOn);
                                        ~PathTracerWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override;

        const char*                     Type() const override { return TypeName(); }
};

template<class MGroup, class PGroup>
class PathTracerBoundaryWork
    : public GPUWorkBatch<PathTracerGlobal, PathTracerLocal, RayAuxPath,
                          MGroup, PGroup, PathLightWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
        bool                            neeOn;
        bool                            misOn;

    protected:
    public:
        // Constrcutors & Destructor
                                        PathTracerBoundaryWork(const GPUMaterialGroupI& mg,
                                                               const GPUPrimitiveGroupI& pg,
                                                               const GPUTransformI* const* t,
                                                               bool neeOn, bool misOn,
                                                               bool emptyPrimitive);
                                        ~PathTracerBoundaryWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override { return 0; }

        const char*                     Type() const override { return TypeName(); }
};

template<class PGroup>
class AmbientOcclusionWork
    : public GPUWorkBatch<AmbientOcclusionGlobal, EmptyState, RayAuxAO,
                          EmptyMat<BasicSurface>, PGroup, AOWork<EmptyMat<BasicSurface>>,
                          PGroup::GetSurfaceFunction>
{
    public:
        static const char*              TypeName() { return TypeNameGen("AO"); }

    protected:
    public:
        // Constrcutors & Destructor
                                        AmbientOcclusionWork(const GPUMaterialGroupI& mg,
                                                             const GPUPrimitiveGroupI& pg,
                                                             const GPUTransformI* const* t);
                                        ~AmbientOcclusionWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override { return 1; }

        const char*                     Type() const override { return TypeName(); }
};

class AmbientOcclusionMissWork
    : public GPUWorkBatch<AmbientOcclusionGlobal, EmptyState, RayAuxAO,
                          EmptyMat<EmptySurface>, GPUPrimitiveEmpty, AOMissWork<EmptyMat<EmptySurface>>,
                          GPUPrimitiveEmpty::GetSurfaceFunction>
{
    public:
        static const char*              TypeName() { return "AOMiss"; }

    private:
    protected:
    public:
        // Constrcutors & Destructor
                                        AmbientOcclusionMissWork(const GPUMaterialGroupI& mg,
                                                                 const GPUPrimitiveGroupI& pg,
                                                                 const GPUTransformI* const* t);
                                        ~AmbientOcclusionMissWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override { return 0; }

        const char*                     Type() const override { return TypeName(); }
};

template<class M, class P>
DirectTracerWork<M, P>::DirectTracerWork(const GPUMaterialGroupI& mg,
                                         const GPUPrimitiveGroupI& pg,
                                         const GPUTransformI* const* t)
    : GPUWorkBatch<DirectTracerGlobal, EmptyState, RayAuxBasic,
                   M, P, BasicWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
{}

template<class M, class P>
PathTracerWork<M, P>::PathTracerWork(const GPUMaterialGroupI& mg,
                                     const GPUPrimitiveGroupI& pg,
                                     const GPUTransformI* const* t,
                                     bool neeOn, bool misOn)
    : GPUWorkBatch<PathTracerGlobal, PathTracerLocal, RayAuxPath,
                   M, P, PathWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
    , neeOn(neeOn)
    , misOn(misOn)
{
    // Populate localData
    localData.emptyPrimitive = false;   
}

template<class M, class P>
uint8_t PathTracerWork<M, P>::OutRayCount() const
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
PathTracerBoundaryWork<M, P>::PathTracerBoundaryWork(const GPUMaterialGroupI& mg,
                                                     const GPUPrimitiveGroupI& pg,
                                                     const GPUTransformI* const* t,
                                                     bool neeOn, bool misOn,
                                                     bool emptyPrimitive)
    : GPUWorkBatch<PathTracerGlobal, PathTracerLocal, RayAuxPath,
                   M, P, PathLightWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
    , neeOn(neeOn)
    , misOn(misOn)
{
    // Populate localData
    localData.emptyPrimitive = emptyPrimitive;
}

template<class P>
AmbientOcclusionWork<P>::AmbientOcclusionWork(const GPUMaterialGroupI& mg,
                                              const GPUPrimitiveGroupI& pg,
                                              const GPUTransformI* const* t)
    : GPUWorkBatch<AmbientOcclusionGlobal, EmptyState, RayAuxAO,
                   EmptyMat<BasicSurface>, P, AOWork<EmptyMat<BasicSurface>>,
                   P::GetSurfaceFunction>(mg, pg, t)
{}

inline AmbientOcclusionMissWork::AmbientOcclusionMissWork(const GPUMaterialGroupI& mg,
                                                          const GPUPrimitiveGroupI& pg,
                                                          const GPUTransformI* const* t)
    : GPUWorkBatch<AmbientOcclusionGlobal, EmptyState, RayAuxAO,
                   EmptyMat<EmptySurface>, GPUPrimitiveEmpty, AOMissWork<EmptyMat<EmptySurface>>,
                   GPUPrimitiveEmpty::GetSurfaceFunction>(mg, pg, t)
{}

// Basic Tracer Work Batches
extern template class DirectTracerWork<BarycentricMat, GPUPrimitiveTriangle>;
extern template class DirectTracerWork<SphericalMat, GPUPrimitiveSphere>;

extern template class DirectTracerWork<NormalRenderMat, GPUPrimitiveTriangle>;
extern template class DirectTracerWork<NormalRenderMat, GPUPrimitiveSphere>;

extern template class DirectTracerWork<LambertMat, GPUPrimitiveTriangle>;
extern template class DirectTracerWork<LambertMat, GPUPrimitiveSphere>;

extern template class DirectTracerWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class DirectTracerWork<UnrealMat, GPUPrimitiveSphere>;

extern template class DirectTracerWork<BoundaryMatConstant, GPUPrimitiveEmpty>;
extern template class DirectTracerWork<BoundaryMatConstant, GPUPrimitiveTriangle>;
extern template class DirectTracerWork<BoundaryMatConstant, GPUPrimitiveSphere>;

extern template class DirectTracerWork<BoundaryMatTextured, GPUPrimitiveTriangle>;
extern template class DirectTracerWork<BoundaryMatTextured, GPUPrimitiveSphere>;

extern template class DirectTracerWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>;

// ===================================================
// Path Tracer Work Batches
extern template class PathTracerWork<LambertCMat, GPUPrimitiveTriangle>;
extern template class PathTracerWork<LambertCMat, GPUPrimitiveSphere>;

extern template class PathTracerWork<ReflectMat, GPUPrimitiveTriangle>;
extern template class PathTracerWork<ReflectMat, GPUPrimitiveSphere>;

extern template class PathTracerWork<RefractMat, GPUPrimitiveTriangle>;
extern template class PathTracerWork<RefractMat, GPUPrimitiveSphere>;

extern template class PathTracerWork<LambertMat, GPUPrimitiveTriangle>;
extern template class PathTracerWork<LambertMat, GPUPrimitiveSphere>;

extern template class PathTracerWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class PathTracerWork<UnrealMat, GPUPrimitiveSphere>;

extern template class PathTracerBoundaryWork<BoundaryMatConstant, GPUPrimitiveEmpty>;
extern template class PathTracerBoundaryWork<BoundaryMatConstant, GPUPrimitiveTriangle>;
extern template class PathTracerBoundaryWork<BoundaryMatConstant, GPUPrimitiveSphere>;

extern template class PathTracerBoundaryWork<BoundaryMatTextured, GPUPrimitiveTriangle>;
extern template class PathTracerBoundaryWork<BoundaryMatTextured, GPUPrimitiveSphere>;
                                             
extern template class PathTracerBoundaryWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>;

// ===================================================
// Ambient Occlusion Work Batches
extern template class AmbientOcclusionWork<GPUPrimitiveTriangle>;
extern template class AmbientOcclusionWork<GPUPrimitiveSphere>;

// ===================================================
using DirectTracerWorkerList = TypeList<DirectTracerWork<BarycentricMat, GPUPrimitiveTriangle>,
                                        DirectTracerWork<SphericalMat, GPUPrimitiveSphere>,
                                        DirectTracerWork<NormalRenderMat, GPUPrimitiveTriangle>,
                                        DirectTracerWork<NormalRenderMat, GPUPrimitiveSphere>,
                                        DirectTracerWork<LambertMat, GPUPrimitiveTriangle>,
                                        DirectTracerWork<LambertMat, GPUPrimitiveSphere>,
                                        DirectTracerWork<UnrealMat, GPUPrimitiveTriangle>,
                                        DirectTracerWork<UnrealMat, GPUPrimitiveSphere>,
                                        DirectTracerWork<BoundaryMatConstant, GPUPrimitiveEmpty>,
                                        DirectTracerWork<BoundaryMatConstant, GPUPrimitiveTriangle>,
                                        DirectTracerWork<BoundaryMatConstant, GPUPrimitiveSphere>,
                                        DirectTracerWork<BoundaryMatTextured, GPUPrimitiveTriangle>,
                                        DirectTracerWork<BoundaryMatTextured, GPUPrimitiveSphere>,
                                        DirectTracerWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>>;
// ===================================================
using PathTracerWorkerList = TypeList<PathTracerWork<LambertCMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<LambertCMat, GPUPrimitiveSphere>,
                                      PathTracerWork<ReflectMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<ReflectMat, GPUPrimitiveSphere>,
                                      PathTracerWork<RefractMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<RefractMat, GPUPrimitiveSphere>,
                                      PathTracerWork<LambertMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<LambertMat, GPUPrimitiveSphere>,
                                      PathTracerWork<UnrealMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<UnrealMat, GPUPrimitiveSphere>>;
// ===================================================
using PathTracerBoundaryWorkerList = TypeList<PathTracerBoundaryWork<BoundaryMatConstant, GPUPrimitiveEmpty>,
                                              PathTracerBoundaryWork<BoundaryMatConstant, GPUPrimitiveTriangle>,
                                              PathTracerBoundaryWork<BoundaryMatConstant, GPUPrimitiveSphere>,
                                              PathTracerBoundaryWork<BoundaryMatTextured, GPUPrimitiveTriangle>,
                                              PathTracerBoundaryWork<BoundaryMatTextured, GPUPrimitiveSphere>,
                                              PathTracerBoundaryWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>>;
// ===================================================
using AmbientOcclusionWorkerList = TypeList<AmbientOcclusionWork<GPUPrimitiveTriangle>,
                                            AmbientOcclusionWork<GPUPrimitiveSphere>,
                                            AmbientOcclusionMissWork>;