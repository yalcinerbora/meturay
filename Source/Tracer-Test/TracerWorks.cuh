#pragma once


#include "Materials/BasicMaterials.cuh"
#include "Materials/SampleMaterials.cuh"
#include "TracerKC.cuh"

#include "TracerLib/WorkPool.h"
#include "TracerLib/GPUWork.cuh"
#include "TracerLib/GPUPrimitiveTriangle.h"
#include "TracerLib/GPUPrimitiveSphere.h"
#include "TracerLib/GPUMaterialLight.cuh"

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
                                                       bool neeOn);
                                        ~PathTracerWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override;
        
};

template<class MGroup, class PGroup,
         SurfaceFunc<MGroup, PGroup> SFunc>
class PathTracerLightWork 
    : public GPUWorkBatch<PathTracerGlobal, PathTracerLocal, RayAuxPath,
                          MGroup, PGroup, SFunc, PathLightWork<MGroup>>
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
                                                            bool neeOn,
                                                            bool emptyPrimitive);
                                        ~PathTracerLightWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override { return 0; }
        
};

template<class MGroup, class PGroup, 
         SurfaceFunc<MGroup, PGroup> SFunc>
class AmbientOcclusionWork 
    : public GPUWorkBatch<AmbientOcclusionGlobal, EmptyState, RayAuxAO,
                          MGroup, PGroup, SFunc, AOWork<MGroup>>
{
    public:
        const char*                     Type() const override { return TypeName(); }

    private:
    protected:
    public:
        // Constrcutors & Destructor
                                        AmbientOcclusionWork(const GPUMaterialGroupI& mg,
                                                            const GPUPrimitiveGroupI& pg);
                                        ~AmbientOcclusionWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override { return 1; }
        
};

template<class MGroup, class PGroup, 
         SurfaceFunc<MGroup, PGroup> SFunc>
class AmbientOcclusionMissWork 
    : public GPUWorkBatch<AmbientOcclusionGlobal, EmptyState, RayAuxAO,
                          MGroup, PGroup, SFunc, AOMissWork<MGroup>>
{
    public:
        const char*                     Type() const override { return TypeName(); }

    private:
    protected:
    public:
        // Constrcutors & Destructor
                                        AmbientOcclusionMissWork(const GPUMaterialGroupI& mg,
                                                                 const GPUPrimitiveGroupI& pg);
                                        ~AmbientOcclusionMissWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override { return 1; }
        
};


template<class M, class P, SurfaceFunc<M, P> S>
PathTracerWork<M, P, S>::PathTracerWork(const GPUMaterialGroupI& mg,
                                        const GPUPrimitiveGroupI& pg,
                                        bool neeOn)
    : GPUWorkBatch(mg, pg)
    , neeOn(neeOn) 
{
    // Populate localData
    localData.emptyPrimitive = false;
    localData.emissiveMaterial = materialGroup.IsEmissiveGroup();
    localData.specularMaterial = materialGroup.IsSpecularGroup();
}

template<class M, class P, SurfaceFunc<M, P> S>
uint8_t PathTracerWork<M,P,S>::OutRayCount() const
{ 
    if(materialGroup.IsSpecularGroup())
        return materialGroup.SampleStrategyCount();
    else if(materialGroup.SampleStrategyCount() != 0)
        return materialGroup.SampleStrategyCount() + ((neeOn) ? 1 : 0);
    return 0;
}

template<class M, class P, SurfaceFunc<M, P> S>
PathTracerLightWork<M, P, S>::PathTracerLightWork(const GPUMaterialGroupI& mg,
                                                  const GPUPrimitiveGroupI& pg,
                                                  bool neeOn,
                                                  bool emptyPrimitive)
    : GPUWorkBatch(mg, pg)
    , neeOn(neeOn) 
{
    // Populate localData
    localData.emptyPrimitive = emptyPrimitive;
    localData.emissiveMaterial = materialGroup.IsEmissiveGroup();
}

template<class M, class P, SurfaceFunc<M, P> S>
AmbientOcclusionWork<M, P, S>::AmbientOcclusionWork(const GPUMaterialGroupI& mg,
                                                    const GPUPrimitiveGroupI& pg)
    : GPUWorkBatch(mg, pg)
{}

template<class M, class P, SurfaceFunc<M, P> S>
AmbientOcclusionMissWork<M, P, S>::AmbientOcclusionMissWork(const GPUMaterialGroupI& mg,
                                                            const GPUPrimitiveGroupI& pg)
    : GPUWorkBatch(mg, pg)
{}

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
// Path Tracer Light Work Batches
extern template class PathTracerLightWork<LightMatConstant,
                                          GPUPrimitiveEmpty,
                                          EmptySurfaceFromEmpty>;
extern template class PathTracerLightWork<LightMatConstant,
                                          GPUPrimitiveTriangle,
                                          EmptySurfaceFromTri>;
extern template class PathTracerLightWork<LightMatConstant,
                                          GPUPrimitiveSphere,
                                          EmptySurfaceFromSphr>;
extern template class PathTracerLightWork<LightMatTextured,
                                          GPUPrimitiveTriangle,
                                          BasicUVSurfaceFromTri>;
extern template class PathTracerLightWork<LightMatTextured,
                                          GPUPrimitiveSphere,
                                          BasicUVSurfaceFromSphr>;
extern template class PathTracerLightWork<LightMatCube,
                                          GPUPrimitiveEmpty,
                                          EmptySurfaceFromEmpty>;
extern template class PathTracerLightWork<LightMatCube,
                                          GPUPrimitiveTriangle,
                                          EmptySurfaceFromTri>;
extern template class PathTracerLightWork<LightMatCube,
                                          GPUPrimitiveSphere,
                                          EmptySurfaceFromSphr>;
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

using PathTracerLightWorkerList = TypeList<PathTracerLightWork<LightMatConstant,
                                                               GPUPrimitiveEmpty,
                                                               EmptySurfaceFromEmpty>,
                                           PathTracerLightWork<LightMatConstant,
                                                               GPUPrimitiveTriangle,
                                                               EmptySurfaceFromTri>,
                                           PathTracerLightWork<LightMatConstant,
                                                               GPUPrimitiveSphere,
                                                               EmptySurfaceFromSphr>,
                                           PathTracerLightWork<LightMatTextured,
                                                               GPUPrimitiveTriangle,
                                                               BasicUVSurfaceFromTri>,
                                           PathTracerLightWork<LightMatTextured,
                                                               GPUPrimitiveSphere,
                                                               BasicUVSurfaceFromSphr>,
                                           PathTracerLightWork<LightMatCube,
                                                               GPUPrimitiveEmpty,
                                                               EmptySurfaceFromEmpty>,
                                           PathTracerLightWork<LightMatCube,
                                                               GPUPrimitiveTriangle,
                                                               EmptySurfaceFromTri>,
                                           PathTracerLightWork<LightMatCube,
                                                               GPUPrimitiveSphere,
                                                               EmptySurfaceFromSphr>>;