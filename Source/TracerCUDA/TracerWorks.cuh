#pragma once

#include "BasicMaterials.cuh"
#include "SampleMaterials.cuh"
#include "UnrealMaterial.cuh"
#include "LambertTexMaterial.cuh"
#include "TracerKC.cuh"

#include "WorkPool.h"
#include "GPUWork.cuh"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"
#include "GPUPrimitiveEmpty.h"
#include "GPUMaterialLight.cuh"

template<class MGroup, class PGroup>
class DirectTracerWork 
    : public GPUWorkBatch<DirectTracerGlobal, EmptyState, RayAuxBasic,
                          MGroup, PGroup, BasicWork<MGroup>, 
                          PGroup::GetSurfaceFunction>
{
    public:
        const char*                     Type() const override { return TypeName(); }

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
        
};

template<class MGroup, class PGroup>
class PathTracerWork 
    : public GPUWorkBatch<PathTracerGlobal, PathTracerLocal, RayAuxPath,
                          MGroup, PGroup, PathWork<MGroup>, 
                          PGroup::GetSurfaceFunction>
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
                                                       const GPUTransformI* const* t,
                                                       bool neeOn);
                                        ~PathTracerWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override;
        
};

template<class MGroup, class PGroup>
class PathTracerLightWork 
    : public GPUWorkBatch<PathTracerGlobal, PathTracerLocal, RayAuxPath,
                          MGroup, PGroup, PathLightWork<MGroup>, 
                          PGroup::GetSurfaceFunction>
{
    public:
        const char*                     Type() const override { return TypeName(); }

    private:
        bool                            neeOn;

    protected:
    public:
        // Constrcutors & Destructor
                                        PathTracerLightWork(const GPUMaterialGroupI& mg,
                                                            const GPUPrimitiveGroupI& pg,
                                                            const GPUTransformI* const* t,
                                                            bool neeOn,
                                                            bool emptyPrimitive);
                                        ~PathTracerLightWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override { return 0; }
        
};

template<class MGroup, class PGroup>
class AmbientOcclusionWork 
    : public GPUWorkBatch<AmbientOcclusionGlobal, EmptyState, RayAuxAO,
                          MGroup, PGroup, AOWork<MGroup>, 
                          PGroup::GetSurfaceFunction>
{
    public:
        const char*                     Type() const override { return TypeName(); }

    private:
    protected:
    public:
        // Constrcutors & Destructor
                                        AmbientOcclusionWork(const GPUMaterialGroupI& mg,
                                                             const GPUPrimitiveGroupI& pg,
                                                             const GPUTransformI* const* t);
                                        ~AmbientOcclusionWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override { return 1; }
        
};

template<class MGroup, class PGroup>
class AmbientOcclusionMissWork 
    : public GPUWorkBatch<AmbientOcclusionGlobal, EmptyState, RayAuxAO,
                          MGroup, PGroup, AOMissWork<MGroup>, 
                          PGroup::GetSurfaceFunction>
{
    public:
        const char*                     Type() const override { return TypeName(); }

    private:
    protected:
    public:
        // Constrcutors & Destructor
                                        AmbientOcclusionMissWork(const GPUMaterialGroupI& mg,
                                                                 const GPUPrimitiveGroupI& pg,
                                                                 const GPUTransformI* const* t);
                                        ~AmbientOcclusionMissWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override { return 1; }
        
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
                                     bool neeOn)
    : GPUWorkBatch<PathTracerGlobal, PathTracerLocal, RayAuxPath,
                   M, P, PathWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
    , neeOn(neeOn) 
{
    // Populate localData
    localData.emptyPrimitive = false;
    localData.emissiveMaterial = materialGroup.IsEmissiveGroup();
    localData.specularMaterial = materialGroup.IsSpecularGroup();
}

template<class M, class P>
uint8_t PathTracerWork<M, P>::OutRayCount() const
{ 
    if(materialGroup.IsSpecularGroup())
        return materialGroup.SampleStrategyCount();
    else if(materialGroup.SampleStrategyCount() != 0)
        return materialGroup.SampleStrategyCount() + ((neeOn) ? 1 : 0);
    return 0;
}

template<class M, class P>
PathTracerLightWork<M, P>::PathTracerLightWork(const GPUMaterialGroupI& mg,
                                               const GPUPrimitiveGroupI& pg,
                                               const GPUTransformI* const* t,
                                               bool neeOn,
                                               bool emptyPrimitive)
    : GPUWorkBatch<PathTracerGlobal, PathTracerLocal, RayAuxPath,
                   M, P, PathLightWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
    , neeOn(neeOn) 
{
    // Populate localData
    localData.emptyPrimitive = emptyPrimitive;
    localData.emissiveMaterial = materialGroup.IsEmissiveGroup();
}

template<class M, class P>
AmbientOcclusionWork<M, P>::AmbientOcclusionWork(const GPUMaterialGroupI& mg,
                                                 const GPUPrimitiveGroupI& pg,
                                                 const GPUTransformI* const* t)
    : GPUWorkBatch<AmbientOcclusionGlobal, EmptyState, RayAuxAO,
                   M, P, AOWork<M>, 
                   P::GetSurfaceFunction>(mg, pg, t)
{}

template<class M, class P>
AmbientOcclusionMissWork<M, P>::AmbientOcclusionMissWork(const GPUMaterialGroupI& mg,
                                                         const GPUPrimitiveGroupI& pg,
                                                         const GPUTransformI* const* t)
    : GPUWorkBatch<AmbientOcclusionGlobal, EmptyState, RayAuxAO,
                   M, P, AOMissWork<M>,
                   P::GetSurfaceFunction>(mg, pg, t)
{}

// Basic Tracer Work Batches
extern template class DirectTracerWork<ConstantMat, GPUPrimitiveEmpty>;
extern template class DirectTracerWork<ConstantMat, GPUPrimitiveTriangle>;
extern template class DirectTracerWork<ConstantMat, GPUPrimitiveSphere>;

extern template class DirectTracerWork<BarycentricMat, GPUPrimitiveTriangle>;
extern template class DirectTracerWork<SphericalMat, GPUPrimitiveSphere>;

extern template class DirectTracerWork<NormalRenderMat, GPUPrimitiveTriangle>;
extern template class DirectTracerWork<NormalRenderMat, GPUPrimitiveSphere>;

extern template class DirectTracerWork<LambertTexMat, GPUPrimitiveTriangle>;
extern template class DirectTracerWork<LambertTexMat, GPUPrimitiveSphere>;

extern template class DirectTracerWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class DirectTracerWork<UnrealMat, GPUPrimitiveSphere>;

// ===================================================
// Path Tracer Work Batches
extern template class PathTracerWork<EmissiveMat, GPUPrimitiveEmpty>;
extern template class PathTracerWork<EmissiveMat, GPUPrimitiveTriangle>;
extern template class PathTracerWork<EmissiveMat, GPUPrimitiveSphere>;

extern template class PathTracerWork<LambertMat, GPUPrimitiveTriangle>;
extern template class PathTracerWork<LambertMat, GPUPrimitiveSphere>;

extern template class PathTracerWork<ReflectMat, GPUPrimitiveTriangle>;
extern template class PathTracerWork<ReflectMat, GPUPrimitiveSphere>;

extern template class PathTracerWork<RefractMat, GPUPrimitiveTriangle>;
extern template class PathTracerWork<RefractMat, GPUPrimitiveSphere>;

extern template class PathTracerWork<LambertTexMat, GPUPrimitiveTriangle>;
extern template class PathTracerWork<LambertTexMat, GPUPrimitiveSphere>;

extern template class PathTracerWork<UnrealMat, GPUPrimitiveTriangle>;
extern template class PathTracerWork<UnrealMat, GPUPrimitiveSphere>;

extern template class PathTracerLightWork<LightMatConstant, GPUPrimitiveEmpty>;
extern template class PathTracerLightWork<LightMatConstant, GPUPrimitiveTriangle>;
extern template class PathTracerLightWork<LightMatConstant, GPUPrimitiveSphere>;

extern template class PathTracerLightWork<LightMatTextured, GPUPrimitiveTriangle>;
extern template class PathTracerLightWork<LightMatTextured, GPUPrimitiveSphere>;

extern template class PathTracerLightWork<LightMatCube, GPUPrimitiveEmpty>;
extern template class PathTracerLightWork<LightMatCube, GPUPrimitiveTriangle>;
extern template class PathTracerLightWork<LightMatCube, GPUPrimitiveSphere>;

// ===================================================
using DirectTracerWorkerList = TypeList<DirectTracerWork<ConstantMat, GPUPrimitiveEmpty>,
                                        DirectTracerWork<ConstantMat, GPUPrimitiveTriangle>,
                                        DirectTracerWork<ConstantMat, GPUPrimitiveSphere>,
                                        DirectTracerWork<BarycentricMat, GPUPrimitiveTriangle>,
                                        DirectTracerWork<SphericalMat, GPUPrimitiveSphere>,
                                        DirectTracerWork<NormalRenderMat, GPUPrimitiveTriangle>,
                                        DirectTracerWork<NormalRenderMat, GPUPrimitiveSphere>,
                                        DirectTracerWork<LambertTexMat, GPUPrimitiveTriangle>,
                                        DirectTracerWork<LambertTexMat, GPUPrimitiveSphere>,
                                        DirectTracerWork<UnrealMat, GPUPrimitiveTriangle>,
                                        DirectTracerWork<UnrealMat, GPUPrimitiveSphere>>;
// ===================================================
using PathTracerWorkerList = TypeList<PathTracerWork<EmissiveMat, GPUPrimitiveEmpty>,
                                      PathTracerWork<EmissiveMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<EmissiveMat, GPUPrimitiveSphere>,
                                      PathTracerWork<LambertMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<LambertMat, GPUPrimitiveSphere>,
                                      PathTracerWork<ReflectMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<ReflectMat, GPUPrimitiveSphere>,
                                      PathTracerWork<RefractMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<RefractMat, GPUPrimitiveSphere>,
                                      PathTracerWork<LambertTexMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<LambertTexMat, GPUPrimitiveSphere>,
                                      PathTracerWork<UnrealMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<UnrealMat, GPUPrimitiveSphere>>;
// ===================================================
using PathTracerLightWorkerList = TypeList<PathTracerLightWork<LightMatConstant, GPUPrimitiveEmpty>,
                                           PathTracerLightWork<LightMatConstant, GPUPrimitiveTriangle>,
                                           PathTracerLightWork<LightMatConstant, GPUPrimitiveSphere>,
                                           PathTracerLightWork<LightMatTextured, GPUPrimitiveTriangle>,
                                           PathTracerLightWork<LightMatTextured, GPUPrimitiveSphere>,
                                           PathTracerLightWork<LightMatCube, GPUPrimitiveEmpty>,
                                           PathTracerLightWork<LightMatCube, GPUPrimitiveTriangle>,
                                           PathTracerLightWork<LightMatCube, GPUPrimitiveSphere>>;