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
#include "DirectTracerKC.cuh"
#include "WorkPool.h"
#include "GPUWork.cuh"

template<class MGroup, class PGroup>
class DirectTracerWork
    : public GPUWorkBatch<DirectTracerGlobalState, 
                          DirectTracerLocalState, RayAuxBasic,
                          MGroup, PGroup, DirectWork<MGroup>,
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

template<class M, class P>
DirectTracerWork<M, P>::DirectTracerWork(const GPUMaterialGroupI& mg,
                                         const GPUPrimitiveGroupI& pg,
                                         const GPUTransformI* const* t)
    : GPUWorkBatch<DirectTracerGlobalState, 
                   DirectTracerLocalState, RayAuxBasic,
                   M, P, DirectWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
{}

// ===================================================
// Direct Tracer Work Batches
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