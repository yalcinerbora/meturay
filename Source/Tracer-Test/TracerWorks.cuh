#pragma once

#include "MetaTracerWork.cuh"
#include "BasicMaterials.cuh"
#include "TracerKC.cuh"
#include "SurfaceStructs.h"
#include "MetaWorkPool.h"

#include "TracerLib/GPUPrimitiveTriangle.h"
#include "TracerLib/GPUPrimitiveSphere.h"


template<class MGroup, class PGroup,
          SurfaceFunc<MGroup, PGroup> SFunc>
class DirectTracerWork 
    : public MetaTracerBatch<BasicTracerGlobal, EmptyState, RayAuxBasic,
                            MGroup, PGroup, SFunc, BasicWork<MGroup>>
{
    public:
        static constexpr const char*    TypeName() { return ""; }
        const char*                     Type() const override { return TypeName(); }

    private:
    protected:
    public:
        // Constrcutors & Destructor
                                        DirectTracerWork(const GPUMaterialGroupI&,
                                                        const GPUPrimitiveGroupI&);
                                        ~DirectTracerWork() = default;

        void                            PreWork() override {}
        // We will not bounce more than once
        uint8_t                         OutRayCount() const override { return 0; }
        
};

template<class MG, class PG,
    SurfaceFunc<MG, PG> SF>
DirectTracerWork<MG, PG, SF>::DirectTracerWork(const GPUMaterialGroupI& mg,
                                              const GPUPrimitiveGroupI& pg)
    : MetaTracerBatch(mg, pg)
{}

// Basic Tracer Work Batches
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

using DirectTracerWorkerList = TypeList<DirectTracerWork<ConstantMat,
                                                         GPUPrimitiveTriangle,
                                                         EmptySurfaceFromTri>,
                                        DirectTracerWork<BarycentricMat,
                                                         GPUPrimitiveTriangle,
                                                         BarySurfaceFromTri>,
                                        DirectTracerWork<ConstantMat,
                                                         GPUPrimitiveSphere,
                                                         EmptySurfaceFromSphr>,
                                        DirectTracerWork<SphericalMat,
                                                         GPUPrimitiveSphere,
                                                         SphrSurfaceFromSphr>>;