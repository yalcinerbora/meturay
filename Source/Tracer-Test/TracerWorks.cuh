#pragma once


#include "BasicMaterials.cuh"
#include "SampleMaterials.cuh"
#include "TracerKC.cuh"
#include "SurfaceStructs.h"

#include "TracerLib/WorkPool.h"
#include "TracerLib/GPUWork.cuh"
#include "TracerLib/GPUPrimitiveTriangle.h"
#include "TracerLib/GPUPrimitiveSphere.h"


template<class MGroup, class PGroup,
         SurfaceFunc<MGroup, PGroup> SFunc>
class DirectTracerWork 
    : public GPUWorkBatch<DirectTracerGlobal, EmptyState, RayAuxBasic,
                          MGroup, PGroup, SFunc, BasicWork<MGroup>>
{
    public:
        const char*                     Type() const override { return TypeName(); }

    private:
    protected:
    public:
        // Constrcutors & Destructor
                                        DirectTracerWork(const GPUMaterialGroupI& mg,
                                                        const GPUPrimitiveGroupI& pg)
                                            : GPUWorkBatch(mg, pg) {}
                                        ~DirectTracerWork() = default;

        void                            GetReady() override {}
        // We will not bounce more than once
        uint8_t                         OutRayCount() const override { return 0; }
        
};

template<class MGroup, class PGroup,
         SurfaceFunc<MGroup, PGroup> SFunc>
class PathTracerWork 
    : public GPUWorkBatch<PathTracerGlobal, PathTracerLocal, RayAuxPath,
                          MGroup, PGroup, SFunc, PathWork<MGroup>>
{
    public:
        const char*                     Type() const override { return TypeName(); }

    private:
        bool                            neeOn;

    protected:
    public:
        // Constrcutors & Destructor
                                        PathTracerWork(const GPUMaterialGroupI& mg,
                                                       const GPUPrimitiveGroupI& pg,
                                                       bool neeOn)
                                            : GPUWorkBatch(mg, pg)
                                            , neeOn(neeOn) {}
                                        ~PathTracerWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override;
        
};

template<class M, class P, SurfaceFunc<M, P> S>
uint8_t PathTracerWork<M,P,S>::OutRayCount() const
{ 
    return materialGroup.SampleStrategyCount() + ((neeOn) ? 1 : 0);
}

// Basic Tracer Work Batches
extern template class DirectTracerWork<ConstantMat,
                                       GPUPrimitiveEmpty,
                                       EmptySurfaceFromEmpty>;

extern template class DirectTracerWork<ConstantMat,
                                       GPUPrimitiveTriangle,
                                       EmptySurfaceFromTri>;

extern template class DirectTracerWork<BarycentricMat,
                                       GPUPrimitiveTriangle,
                                       BarySurfaceFromTri>;
//
extern template class DirectTracerWork<ConstantMat,
                                       GPUPrimitiveSphere,
                                       EmptySurfaceFromSphr>;

extern template class DirectTracerWork<SphericalMat,
                                       GPUPrimitiveSphere,
                                       SphrSurfaceFromSphr>;
// ===================================================
// Path Tracer Work Batches
extern template class PathTracerWork<EmissiveMat,
                                     GPUPrimitiveEmpty,
                                     EmptySurfaceFromEmpty>;

extern template class PathTracerWork<EmissiveMat,
                                     GPUPrimitiveTriangle,
                                     EmptySurfaceFromTri>;

extern template class PathTracerWork<LambertMat,
                                     GPUPrimitiveTriangle,
                                     BasicSurfaceFromTri>;

extern template class PathTracerWork<ReflectMat,
                                     GPUPrimitiveTriangle,
                                     BasicSurfaceFromTri>;

extern template class PathTracerWork<RefractMat,
                                     GPUPrimitiveTriangle,
                                     BasicSurfaceFromTri>;
//
extern template class PathTracerWork<EmissiveMat,
                                     GPUPrimitiveSphere,
                                     EmptySurfaceFromSphr>;

extern template class PathTracerWork<LambertMat,
                                     GPUPrimitiveSphere,
                                     BasicSurfaceFromSphr>;

extern template class PathTracerWork<ReflectMat,
                                     GPUPrimitiveSphere,
                                     BasicSurfaceFromSphr>;

extern template class PathTracerWork<RefractMat,
                                     GPUPrimitiveSphere,
                                     BasicSurfaceFromSphr>;
// ===================================================

using DirectTracerWorkerList = TypeList<DirectTracerWork<ConstantMat,
                                                         GPUPrimitiveTriangle,
                                                         EmptySurfaceFromTri>,
                                        DirectTracerWork<ConstantMat,
                                                         GPUPrimitiveEmpty,
                                                         EmptySurfaceFromEmpty>,
                                        DirectTracerWork<BarycentricMat,
                                                         GPUPrimitiveTriangle,
                                                         BarySurfaceFromTri>,
                                        DirectTracerWork<ConstantMat,
                                                         GPUPrimitiveSphere,
                                                         EmptySurfaceFromSphr>,
                                        DirectTracerWork<SphericalMat,
                                                         GPUPrimitiveSphere,
                                                         SphrSurfaceFromSphr>>;

using PathTracerWorkerList = TypeList<PathTracerWork<EmissiveMat,
                                                     GPUPrimitiveEmpty,
                                                     EmptySurfaceFromEmpty>,
                                      PathTracerWork<EmissiveMat,
                                                     GPUPrimitiveTriangle,
                                                     EmptySurfaceFromTri>,
                                      PathTracerWork<LambertMat,
                                                     GPUPrimitiveTriangle,
                                                     BasicSurfaceFromTri>,
                                      PathTracerWork<ReflectMat,
                                                     GPUPrimitiveTriangle,
                                                     BasicSurfaceFromTri>,
                                      PathTracerWork<RefractMat,
                                                     GPUPrimitiveTriangle,
                                                     BasicSurfaceFromTri>,
                                      PathTracerWork<EmissiveMat,
                                                     GPUPrimitiveSphere,
                                                     EmptySurfaceFromSphr>,                                
                                      PathTracerWork<LambertMat,
                                                     GPUPrimitiveSphere,
                                                     BasicSurfaceFromSphr>,
                                      PathTracerWork<ReflectMat,
                                                     GPUPrimitiveSphere,
                                                     BasicSurfaceFromSphr>,
                                      PathTracerWork<RefractMat,
                                                     GPUPrimitiveSphere,
                                                     BasicSurfaceFromSphr>>;