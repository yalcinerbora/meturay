#pragma once

// Primitives
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"
#include "GPUPrimitiveEmpty.h"
// Materials
#include "EmptyMaterial.cuh"
// Misc
#include "AOTracerKC.cuh"
#include "WorkPool.h"
#include "GPUWork.cuh"

template<class PGroup>
class AmbientOcclusionWork
    : public GPUWorkBatch<AmbientOcclusionGlobalState, 
                          AmbientOcclusionLocalState, RayAuxAO,
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
    : public GPUWorkBatch<AmbientOcclusionGlobalState, 
                          AmbientOcclusionLocalState, RayAuxAO,
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


template<class P>
AmbientOcclusionWork<P>::AmbientOcclusionWork(const GPUMaterialGroupI& mg,
                                              const GPUPrimitiveGroupI& pg,
                                              const GPUTransformI* const* t)
    : GPUWorkBatch<AmbientOcclusionGlobalState,
                   AmbientOcclusionLocalState, RayAuxAO,
                   EmptyMat<BasicSurface>, P, AOWork<EmptyMat<BasicSurface>>,
                   P::GetSurfaceFunction>(mg, pg, t)
{}

inline AmbientOcclusionMissWork::AmbientOcclusionMissWork(const GPUMaterialGroupI& mg,
                                                          const GPUPrimitiveGroupI& pg,
                                                          const GPUTransformI* const* t)
    : GPUWorkBatch<AmbientOcclusionGlobalState, 
                   AmbientOcclusionLocalState, RayAuxAO,
                   EmptyMat<EmptySurface>, GPUPrimitiveEmpty, AOMissWork<EmptyMat<EmptySurface>>,
                   GPUPrimitiveEmpty::GetSurfaceFunction>(mg, pg, t)
{}

// ===================================================
// Ambient Occlusion Work Batches
extern template class AmbientOcclusionWork<GPUPrimitiveTriangle>;
extern template class AmbientOcclusionWork<GPUPrimitiveSphere>;
// ===================================================
using AmbientOcclusionWorkerList = TypeList<AmbientOcclusionWork<GPUPrimitiveTriangle>,
                                            AmbientOcclusionWork<GPUPrimitiveSphere>,
                                            AmbientOcclusionMissWork>;