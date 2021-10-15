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
#include "GPULightSkySphere.cuh"
#include "GPULightNull.cuh"
// Misc
#include "DirectTracerKC.cuh"
#include "WorkPool.h"
#include "GPUWork.cuh"

template<class EGroup>
class DirectTracerBoundaryWork final
    : public GPUBoundaryWorkBatch<DirectTracerGlobalState,
                                  DirectTracerLocalState, RayAuxBasic,
                                  EGroup, DirectBoundaryWork<EGroup>>
{
    private:
        bool                neeOn;
        bool                misOn;

        using Base = GPUBoundaryWorkBatch<DirectTracerGlobalState,
                                          DirectTracerLocalState, RayAuxBasic,
                                          EGroup, DirectBoundaryWork<EGroup>>;

    protected:
    public:
        // Constrcutors & Destructor
                                DirectTracerBoundaryWork(const CPUEndpointGroupI& eg,
                                                         const GPUTransformI* const* t);
                                ~DirectTracerBoundaryWork() = default;

        void                    GetReady() override {}
        uint8_t                 OutRayCount() const override { return 0; }

        const char*             Type() const override { return Base::TypeName(); }
};

template<class MGroup, class PGroup>
class DirectTracerFurnaceWork final
    : public GPUWorkBatch<DirectTracerGlobalState,
                          DirectTracerLocalState, RayAuxBasic,
                          MGroup, PGroup, DirectFurnaceWork<MGroup>,
                          PGroup::GetSurfaceFunction>
{
    private:
        using Base = GPUWorkBatch<DirectTracerGlobalState,
                                  DirectTracerLocalState, RayAuxBasic,
                                  MGroup, PGroup, DirectFurnaceWork<MGroup>,
                                  PGroup::GetSurfaceFunction>;

    protected:
    public:
        // Constrcutors & Destructor
                                        DirectTracerFurnaceWork(const GPUMaterialGroupI& mg,
                                                                const GPUPrimitiveGroupI& pg,
                                                                const GPUTransformI* const* t);
                                        ~DirectTracerFurnaceWork() = default;

        void                            GetReady() override {}
        // We will not bounce more than once
        uint8_t                         OutRayCount() const override { return 0; }

        const char*                     Type() const override { return Base::TypeName(); }
};

class DirectTracerPositionWork final
    : public GPUWorkBatch<DirectTracerPositionGlobalState,
                          DirectTracerLocalState, RayAuxBasic,
                          EmptyMat<EmptySurface>, GPUPrimitiveEmpty, DirectPositionWork,
                          GPUPrimitiveEmpty::GetSurfaceFunction>
{
    private:
        using Base = GPUWorkBatch<DirectTracerPositionGlobalState,
                                  DirectTracerLocalState, RayAuxBasic,
                                  EmptyMat<EmptySurface>, GPUPrimitiveEmpty, DirectPositionWork,
                                  GPUPrimitiveEmpty::GetSurfaceFunction>;

    public:
        static const char*              TypeName() { return "DirectPosition"; }

    protected:
    public:
        // Constrcutors & Destructor
                                        DirectTracerPositionWork(const GPUMaterialGroupI& mg,
                                                                 const GPUPrimitiveGroupI& pg,
                                                                 const GPUTransformI* const* t);
                                        ~DirectTracerPositionWork() = default;

        void                            GetReady() override {}
        // We will not bounce more than once
        uint8_t                         OutRayCount() const override { return 0; }

        const char*                     Type() const override { return Base::TypeName(); }
};

template <class PGroup>
class DirectTracerNormalWork final
    : public GPUWorkBatch<DirectTracerGlobalState,
                          DirectTracerLocalState, RayAuxBasic,
                          NormalRenderMat, PGroup, DirectNormalWork,
                          PGroup::GetSurfaceFunction>
{

    public:
        static const char*              TypeName() { return Base::TypeNameGen("DirectNormal"); }


    private:
        using Base = GPUWorkBatch<DirectTracerGlobalState,
                                  DirectTracerLocalState, RayAuxBasic,
                                  NormalRenderMat, PGroup, DirectNormalWork,
                                  PGroup::GetSurfaceFunction>;

    protected:
    public:
        // Constrcutors & Destructor
                                        DirectTracerNormalWork(const GPUMaterialGroupI& mg,
                                                               const GPUPrimitiveGroupI& pg,
                                                               const GPUTransformI* const* t);
                                        ~DirectTracerNormalWork() = default;

        void                            GetReady() override {}
        // We will not bounce more than once
        uint8_t                         OutRayCount() const override { return 0; }

        const char*                     Type() const override { return Base::TypeName(); }
};

template<class E>
DirectTracerBoundaryWork<E>::DirectTracerBoundaryWork(const CPUEndpointGroupI& eg,
                                                      const GPUTransformI* const* t)
    : Base(eg, t)
{}

template<class M, class P>
DirectTracerFurnaceWork<M, P>::DirectTracerFurnaceWork(const GPUMaterialGroupI& mg,
                                                       const GPUPrimitiveGroupI& pg,
                                                       const GPUTransformI* const* t)
    : GPUWorkBatch<DirectTracerGlobalState,
                   DirectTracerLocalState, RayAuxBasic,
                   M, P, DirectFurnaceWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
{}

inline
DirectTracerPositionWork::DirectTracerPositionWork(const GPUMaterialGroupI& mg,
                                                   const GPUPrimitiveGroupI& pg,
                                                   const GPUTransformI* const* t)
    : Base(mg, pg, t)
{}

template <class P>
DirectTracerNormalWork<P>::DirectTracerNormalWork(const GPUMaterialGroupI& mg,
                                                  const GPUPrimitiveGroupI& pg,
                                                  const GPUTransformI* const* t)
    : Base(mg, pg, t)
{}


// ===================================================
// Direct Tracer Boundary Work Batches
extern template class DirectTracerBoundaryWork<CPULightGroupNull>;
extern template class DirectTracerBoundaryWork<CPULightGroupSkySphere>;
// ===================================================
// Direct Tracer Work Batches
extern template class DirectTracerFurnaceWork<BarycentricMat, GPUPrimitiveTriangle>;
extern template class DirectTracerFurnaceWork<SphericalMat, GPUPrimitiveSphere>;

extern template class DirectTracerFurnaceWork<NormalRenderMat, GPUPrimitiveTriangle>;
extern template class DirectTracerFurnaceWork<NormalRenderMat, GPUPrimitiveSphere>;

extern template class DirectTracerFurnaceWork<LambertMat, GPUPrimitiveTriangle>;
extern template class DirectTracerFurnaceWork<LambertMat, GPUPrimitiveSphere>;

extern template class DirectTracerFurnaceWork<LambertCMat, GPUPrimitiveTriangle>;
extern template class DirectTracerFurnaceWork<LambertCMat, GPUPrimitiveSphere>;

extern template class DirectTracerFurnaceWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class DirectTracerFurnaceWork<UnrealMat, GPUPrimitiveSphere>;
// ===================================================
extern template class DirectTracerNormalWork<GPUPrimitiveEmpty>;
extern template class DirectTracerNormalWork<GPUPrimitiveTriangle>;
extern template class DirectTracerNormalWork<GPUPrimitiveSphere>;
// ===================================================
using DirectTracerBoundaryWorkerList = TypeList<DirectTracerBoundaryWork<CPULightGroupNull>,
                                                DirectTracerBoundaryWork<CPULightGroupSkySphere>>;
using DirectTracerFurnaceWorkerList = TypeList<DirectTracerFurnaceWork<BarycentricMat, GPUPrimitiveTriangle>,
                                               DirectTracerFurnaceWork<SphericalMat, GPUPrimitiveSphere>,
                                               DirectTracerFurnaceWork<NormalRenderMat, GPUPrimitiveTriangle>,
                                               DirectTracerFurnaceWork<NormalRenderMat, GPUPrimitiveSphere>,
                                               DirectTracerFurnaceWork<LambertMat, GPUPrimitiveTriangle>,
                                               DirectTracerFurnaceWork<LambertMat, GPUPrimitiveSphere>,
                                               DirectTracerFurnaceWork<LambertCMat, GPUPrimitiveTriangle>,
                                               DirectTracerFurnaceWork<LambertCMat, GPUPrimitiveSphere>,
                                               DirectTracerFurnaceWork<UnrealMat, GPUPrimitiveTriangle>,
                                               DirectTracerFurnaceWork<UnrealMat, GPUPrimitiveSphere>>;
using DirectTracerNormalWorkerList = TypeList<DirectTracerNormalWork<GPUPrimitiveEmpty>,
                                              DirectTracerNormalWork<GPUPrimitiveTriangle>,
                                              DirectTracerNormalWork<GPUPrimitiveSphere>>;
// ===================================================
using DirectTracerPositionWorkerList = TypeList<DirectTracerPositionWork>;