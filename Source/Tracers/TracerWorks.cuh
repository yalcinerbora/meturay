#pragma once


#include "Materials/BasicMaterials.cuh"
#include "Materials/SampleMaterials.cuh"
#include "TracerKC.cuh"

#include "TracerLib/WorkPool.h"
#include "TracerLib/GPUWork.cuh"
#include "TracerLib/GPUPrimitiveTriangle.h"
#include "TracerLib/GPUPrimitiveSphere.h"
#include "TracerLib/GPUPrimitiveEmpty.h"
#include "TracerLib/GPUMaterialLight.cuh"

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
                                                         const GPUTransformI* const* t)
                                            : GPUWorkBatch(mg, pg, t) {}
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
                                                                 const GPUPrimitiveGroupI& pg);
                                        ~AmbientOcclusionMissWork() = default;

        void                            GetReady() override {}
        uint8_t                         OutRayCount() const override { return 1; }
        
};


template<class M, class P>
PathTracerWork<M, P>::PathTracerWork(const GPUMaterialGroupI& mg,
                                     const GPUPrimitiveGroupI& pg,
                                     const GPUTransformI* const* t,
                                     bool neeOn)
    : GPUWorkBatch(mg, pg, t)
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
    : GPUWorkBatch(mg, pg, t)
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
    : GPUWorkBatch(mg, pg, t)
{}

template<class M, class P>
AmbientOcclusionMissWork<M, P>::AmbientOcclusionMissWork(const GPUMaterialGroupI& mg,
                                                         const GPUPrimitiveGroupI& pg)
    : GPUWorkBatch(mg, pg)
{}

// Basic Tracer Work Batches
extern template class DirectTracerWork<ConstantMat, GPUPrimitiveEmpty>;
extern template class DirectTracerWork<ConstantMat, GPUPrimitiveTriangle>;
extern template class DirectTracerWork<ConstantMat, GPUPrimitiveSphere>;

extern template class DirectTracerWork<BarycentricMat, GPUPrimitiveTriangle>;
extern template class DirectTracerWork<SphericalMat, GPUPrimitiveSphere>;
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
                                        DirectTracerWork<SphericalMat, GPUPrimitiveSphere>>;

// ===================================================
using PathTracerWorkerList = TypeList<PathTracerWork<EmissiveMat, GPUPrimitiveEmpty>,
                                      PathTracerWork<EmissiveMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<EmissiveMat, GPUPrimitiveSphere>,
                                      PathTracerWork<LambertMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<LambertMat, GPUPrimitiveSphere>,
                                      PathTracerWork<ReflectMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<ReflectMat, GPUPrimitiveSphere>,
                                      PathTracerWork<RefractMat, GPUPrimitiveTriangle>,
                                      PathTracerWork<RefractMat, GPUPrimitiveSphere>>;
// ===================================================
using PathTracerLightWorkerList = TypeList<PathTracerLightWork<LightMatConstant, GPUPrimitiveEmpty>,
                                           PathTracerLightWork<LightMatConstant, GPUPrimitiveTriangle>,
                                           PathTracerLightWork<LightMatConstant, GPUPrimitiveSphere>,
                                           PathTracerLightWork<LightMatTextured, GPUPrimitiveTriangle>,
                                           PathTracerLightWork<LightMatTextured, GPUPrimitiveSphere>,
                                           PathTracerLightWork<LightMatCube, GPUPrimitiveEmpty>,
                                           PathTracerLightWork<LightMatCube, GPUPrimitiveTriangle>,
                                           PathTracerLightWork<LightMatCube, GPUPrimitiveSphere>>;

//// Basic Tracer Work Batches
//extern template class DirectTracerWork<ConstantMat,
//                                       GPUPrimitiveEmpty,
//                                       EmptySurfaceFromAny<GPUPrimitiveEmpty>>;
//
//extern template class DirectTracerWork<ConstantMat,
//                                       GPUPrimitiveTriangle,
//                                       EmptySurfaceFromAny<GPUPrimitiveTriangle>>;
//
//extern template class DirectTracerWork<BarycentricMat,
//                                       GPUPrimitiveTriangle,
//                                       BarySurfaceFromTri>;
////
//extern template class DirectTracerWork<ConstantMat,
//                                       GPUPrimitiveSphere,
//                                       EmptySurfaceFromAny<GPUPrimitiveSphere>>;
//
//extern template class DirectTracerWork<SphericalMat,
//                                       GPUPrimitiveSphere,
//                                       SphrSurfaceFromSphr>;
//// ===================================================
//// Path Tracer Work Batches
//extern template class PathTracerWork<EmissiveMat,
//                                     GPUPrimitiveEmpty,
//                                     EmptySurfaceFromAny<GPUPrimitiveEmpty>>;
//
//extern template class PathTracerWork<EmissiveMat,
//                                     GPUPrimitiveTriangle,
//                                     EmptySurfaceFromAny<GPUPrimitiveTriangle>>;
//
//extern template class PathTracerWork<LambertMat,
//                                     GPUPrimitiveTriangle,
//                                     BasicSurfaceFromTri>;
//
//extern template class PathTracerWork<ReflectMat,
//                                     GPUPrimitiveTriangle,
//                                     BasicSurfaceFromTri>;
//
//extern template class PathTracerWork<RefractMat,
//                                     GPUPrimitiveTriangle,
//                                     BasicSurfaceFromTri>;
////
//extern template class PathTracerWork<EmissiveMat,
//                                     GPUPrimitiveSphere,
//                                     EmptySurfaceFromAny<GPUPrimitiveSphere>>;
//
//extern template class PathTracerWork<LambertMat,
//                                     GPUPrimitiveSphere,
//                                     BasicSurfaceFromSphr>;
//
//extern template class PathTracerWork<ReflectMat,
//                                     GPUPrimitiveSphere,
//                                     BasicSurfaceFromSphr>;
//
//extern template class PathTracerWork<RefractMat,
//                                     GPUPrimitiveSphere,
//                                     BasicSurfaceFromSphr>;
//// ===================================================
//// Path Tracer Light Work Batches
//extern template class PathTracerLightWork<LightMatConstant,
//                                          GPUPrimitiveEmpty,
//                                          EmptySurfaceFromAny<GPUPrimitiveEmpty>>;
//extern template class PathTracerLightWork<LightMatConstant,
//                                          GPUPrimitiveTriangle,
//                                          EmptySurfaceFromAny<GPUPrimitiveTriangle>>;
//extern template class PathTracerLightWork<LightMatConstant,
//                                          GPUPrimitiveSphere,
//                                          EmptySurfaceFromAny<GPUPrimitiveSphere>>;
//extern template class PathTracerLightWork<LightMatTextured,
//                                          GPUPrimitiveTriangle,
//                                          UVSurfaceFromTri>;
//extern template class PathTracerLightWork<LightMatTextured,
//                                          GPUPrimitiveSphere,
//                                          UVSurfaceFromSphr>;
//extern template class PathTracerLightWork<LightMatCube,
//                                          GPUPrimitiveEmpty,
//                                          EmptySurfaceFromAny<GPUPrimitiveEmpty>>;
//extern template class PathTracerLightWork<LightMatCube,
//                                          GPUPrimitiveTriangle,
//                                          EmptySurfaceFromAny<GPUPrimitiveTriangle>>;
//extern template class PathTracerLightWork<LightMatCube,
//                                          GPUPrimitiveSphere,
//                                          EmptySurfaceFromAny<GPUPrimitiveSphere>>;
//// ===================================================
//
//using DirectTracerWorkerList = TypeList<DirectTracerWork<ConstantMat,
//                                                         GPUPrimitiveTriangle,
//                                                         EmptySurfaceFromAny<GPUPrimitiveTriangle>>,
//                                        DirectTracerWork<ConstantMat,
//                                                         GPUPrimitiveEmpty,
//                                                         EmptySurfaceFromAny<GPUPrimitiveEmpty>>,
//                                        DirectTracerWork<BarycentricMat,
//                                                         GPUPrimitiveTriangle,
//                                                         BarySurfaceFromTri>,
//                                        DirectTracerWork<ConstantMat,
//                                                         GPUPrimitiveSphere,
//                                                         EmptySurfaceFromAny<GPUPrimitiveSphere>>,
//                                        DirectTracerWork<SphericalMat,
//                                                         GPUPrimitiveSphere,
//                                                         SphrSurfaceFromSphr>>;
//
//using PathTracerWorkerList = TypeList<PathTracerWork<EmissiveMat,
//                                                     GPUPrimitiveEmpty,
//                                                     EmptySurfaceFromAny<GPUPrimitiveEmpty>>,
//                                      PathTracerWork<EmissiveMat,
//                                                     GPUPrimitiveTriangle,
//                                                     EmptySurfaceFromAny<GPUPrimitiveTriangle>>,
//                                      PathTracerWork<LambertMat,
//                                                     GPUPrimitiveTriangle,
//                                                     BasicSurfaceFromTri>,
//                                      PathTracerWork<ReflectMat,
//                                                     GPUPrimitiveTriangle,
//                                                     BasicSurfaceFromTri>,
//                                      PathTracerWork<RefractMat,
//                                                     GPUPrimitiveTriangle,
//                                                     BasicSurfaceFromTri>,
//                                      PathTracerWork<EmissiveMat,
//                                                     GPUPrimitiveSphere,
//                                                     EmptySurfaceFromAny<GPUPrimitiveSphere>>,
//                                      PathTracerWork<LambertMat,
//                                                     GPUPrimitiveSphere,
//                                                     BasicSurfaceFromSphr>,
//                                      PathTracerWork<ReflectMat,
//                                                     GPUPrimitiveSphere,
//                                                     BasicSurfaceFromSphr>,
//                                      PathTracerWork<RefractMat,
//                                                     GPUPrimitiveSphere,
//                                                     BasicSurfaceFromSphr>>;
//
//using PathTracerLightWorkerList = TypeList<PathTracerLightWork<LightMatConstant,
//                                                               GPUPrimitiveEmpty,
//                                                               EmptySurfaceFromAny<GPUPrimitiveEmpty>>,
//                                           PathTracerLightWork<LightMatConstant,
//                                                               GPUPrimitiveTriangle,
//                                                               EmptySurfaceFromAny<GPUPrimitiveTriangle>>,
//                                           PathTracerLightWork<LightMatConstant,
//                                                               GPUPrimitiveSphere,
//                                                               EmptySurfaceFromAny<GPUPrimitiveSphere>>,
//                                           PathTracerLightWork<LightMatTextured,
//                                                               GPUPrimitiveTriangle,
//                                                               UVSurfaceFromTri>,
//                                           PathTracerLightWork<LightMatTextured,
//                                                               GPUPrimitiveSphere,
//                                                               UVSurfaceFromSphr>,
//                                           PathTracerLightWork<LightMatCube,
//                                                               GPUPrimitiveEmpty,
//                                                               EmptySurfaceFromAny<GPUPrimitiveEmpty>>,
//                                           PathTracerLightWork<LightMatCube,
//                                                               GPUPrimitiveTriangle,
//                                                               EmptySurfaceFromAny<GPUPrimitiveTriangle>>,
//                                           PathTracerLightWork<LightMatCube,
//                                                               GPUPrimitiveSphere,
//                                                               EmptySurfaceFromAny<GPUPrimitiveSphere>>>;