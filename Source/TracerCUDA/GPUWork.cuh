#pragma once

#include "GPUWorkI.h"
#include "Random.cuh"
#include "GPUPrimitiveP.cuh"
#include "GPUMaterialP.cuh"
#include "RNGMemory.h"
#include "MangledNames.h"
#include "WorkKernels.cuh"
#include "EndpointFinder.cuh"
#include "GPUEndpointI.h"

#include "RayLib/TracerError.h"

// Meta Tracer Code
// With custom global Data

// CUDA complains when generator function
// is called as a static member function
// instead we supply it as a template parameter
template <class S, class H, class D>
using SurfaceFuncGenerator = SurfaceFunc<S, H, D>(*)();

// Material/Primitive invaritant part of the code
template<class GlobalData, class RayData>
class GPUWorkBatchD : public GPUWorkBatchI
{
    protected:
        // Ray Auxiliary Input and output pointers
        // which are global (not local)
        const RayData*  dAuxInGlobal = nullptr;
        RayData*        dAuxOutLocal = nullptr;

        // GPU Friendly Struct which will be directly passed to the kernel call
        GlobalData      globalData;

    public:
        // Constructors & Destructor
                        GPUWorkBatchD() = default;
                        ~GPUWorkBatchD() = default;

        void            SetGlobalData(const GlobalData&);
        void            SetRayDataPtrs(RayData* rayDataOutLocal,
                                       const RayData* rayDataInGlobal);
};

template<class GlobalData, class LocalData, class RayData,
         class MGroup, class PGroup,
         WorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
class GPUWorkBatch
    : public GPUWorkBatchD<GlobalData, RayData>
{
    public:
        static const char*              TypeNameGen(const char* mgOverride = nullptr,
                                                    const char* pgOverride = nullptr);
        static const char*              TypeName() {return TypeNameGen();}

    private:
        using SF =                      SurfaceFunc<typename MGroup::Surface,
                                                    typename PGroup::HitData,
                                                    typename PGroup::PrimitiveData>;
        using WF =                      WorkFunc<GlobalData, LocalData,
                                                 RayData, MGroup>;

        static constexpr SF             SurfF = SGen();
        static constexpr WF             WorkF = WFunc;

    protected:
        const MGroup&                   materialGroup;
        const PGroup&                   primitiveGroup;

        const GPUTransformI* const*     dTransforms;

        // Per-Bathch Data
        LocalData                       localData;

    public:
        // Constrcutors & Destructor
                                        GPUWorkBatch(const GPUMaterialGroupI& mg,
                                                     const GPUPrimitiveGroupI& pg,
                                                     const GPUTransformI* const* t);
                                        ~GPUWorkBatch() = default;

        void                            Work(// Output
                                             HitKey* dBoundMatOut,
                                             RayGMem* dRayOut,
                                             //  Input
                                             const RayGMem* dRayIn,
                                             const PrimitiveId* dPrimitiveIds,
                                             const TransformId* dTransformIds,
                                             const HitStructPtr dHitStructs,
                                             // Ids
                                             const HitKey* dMatIds,
                                             const RayId* dRayIds,
                                             //
                                             const uint32_t rayCount,
                                             RNGMemory& rngMem) override;

        const GPUPrimitiveGroupI&       PrimitiveGroup() const override { return primitiveGroup; }
        const GPUMaterialGroupI&        MaterialGroup() const override { return materialGroup; }
};

template<class GlobalData, class LocalData, class RayData,
         class MGroup, class PGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
class GPUBoundaryWorkBatch 
    : public GPUWorkBatchD<GlobalData, RayData>
{
    public:
        static const char*              TypeNameGen(const char* mgOverride = nullptr,
                                                    const char* pgOverride = nullptr);
        static const char*              TypeName() {return TypeNameGen();}

    private:
        using SF =                      SurfaceFunc<typename MGroup::Surface,
                                                    typename PGroup::HitData,
                                                    typename PGroup::PrimitiveData>;
        using WF =                      BoundaryWorkFunc<GlobalData, LocalData,
                                                         RayData, MGroup>;

        static constexpr SF             SurfF = SGen();
        static constexpr WF             WorkF = WFunc;

    protected:
        const MGroup&                   materialGroup;
        const PGroup&                   primitiveGroup;

        const GPUTransformI* const*     dTransforms;

        // Endpoint Finder
        DeviceMemory                    endPointFinderMemory;
        EndpointFinder                  endpointFinder;

        // Per-Bathch Data
        LocalData                       localData;

    public:
        // Constrcutors & Destructor
                                        GPUBoundaryWorkBatch(const GPUMaterialGroupI& mg,
                                                             const GPUPrimitiveGroupI& pg,
                                                             const GPUTransformI* const* t);
                                        ~GPUBoundaryWorkBatch() = default;

        TracerError                    GenerateEndpointFinder(const GPUEndpointI* const* sceneEndPoints);

        void                            Work(// Output
                                             HitKey* dBoundMatOut,
                                             RayGMem* dRayOut,
                                             //  Input
                                             const RayGMem* dRayIn,
                                             const PrimitiveId* dPrimitiveIds,
                                             const TransformId* dTransformIds,
                                             const HitStructPtr dHitStructs,
                                             // Ids
                                             const HitKey* dMatIds,
                                             const RayId* dRayIds,
                                             //
                                             const uint32_t rayCount,
                                             RNGMemory& rngMem) override;

        const GPUPrimitiveGroupI&       PrimitiveGroup() const override { return primitiveGroup; }
        const GPUMaterialGroupI&        MaterialGroup() const override { return materialGroup; }
};

template<class GD, class RD>
inline void GPUWorkBatchD<GD, RD>::SetGlobalData(const GD& d)
{
    globalData = d;
}

template<class GD, class RD>
void GPUWorkBatchD<GD, RD>::SetRayDataPtrs(RD* dRDOut,
                                           const RD* dRDInGlobal)
{
    dAuxInGlobal = dRDInGlobal;
    dAuxOutLocal = dRDOut;
}

template<class GlobalData, class LocalData, class RayData,
         class MGroup, class PGroup,
         WorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline const char* GPUWorkBatch<GlobalData, LocalData, RayData,
                                MGroup, PGroup, WFunc, SGen>::TypeNameGen(const char* mgOverride,
                                                                          const char* pgOverride)
{
    const char* pgName = PGroup::TypeName();
    const char* mgName = MGroup::TypeName();
    if(pgOverride) pgName = pgOverride;
    if(mgOverride) mgName = mgOverride;

    static std::string typeName = MangledNames::WorkBatch(pgName,
                                                          mgName);
    return typeName.c_str();
}

template<class GlobalData, class LocalData, class RayData,
         class MGroup, class PGroup,
         WorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
GPUWorkBatch<GlobalData, LocalData, RayData,
             MGroup, PGroup, WFunc, SGen>::GPUWorkBatch(const GPUMaterialGroupI& mg,
                                                        const GPUPrimitiveGroupI& pg,
                                                        const GPUTransformI* const* t)
    : materialGroup(static_cast<const MGroup&>(mg))
    , primitiveGroup(static_cast<const PGroup&>(pg))
    , dTransforms(t)
{}

template<class GlobalData, class LocalData, class RayData, class MGroup, class PGroup,
         WorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
void GPUWorkBatch<GlobalData, LocalData, RayData,
                  MGroup, PGroup, WFunc, SGen>::Work(// Output
                                                     HitKey* dBoundMatOut,
                                                     RayGMem* dRayOut,
                                                     //  Input
                                                     const RayGMem* dRayIn,
                                                     const PrimitiveId* dPrimitiveIds,
                                                     const TransformId* dTransformIds,
                                                     const HitStructPtr dHitStructs,
                                                     // Ids
                                                     const HitKey* dMatIds,
                                                     const RayId* dRayIds,
                                                     //
                                                     const uint32_t rayCount,
                                                     RNGMemory& rngMem)
{
    // Do Pre-work (initialize local data etc.)
    this->GetReady();

    using PrimitiveGroup = PGroup;
    using PrimitiveData = typename PGroup::PrimitiveData;
    using HitData = typename PGroup::HitData;
    using MaterialData = typename MGroup::Data;
    using MaterialSurface = typename MGroup::Surface;

    // Get Data
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
    const MaterialData matData = MatDataAccessor::Data(materialGroup);

    const uint32_t outRayCount = this->OutRayCount();

    const CudaGPU& gpu = materialGroup.GPU();
    gpu.AsyncGridStrideKC_X
    (
        0,
        rayCount,
        //
        KCWork<GlobalData, LocalData, RayData, PGroup,
               MGroup, WorkF, SurfF>,
        // Args
        // Output
        dBoundMatOut,
        dRayOut,
        this->dAuxOutLocal,
        outRayCount,
        // Input
        dRayIn,
        this->dAuxInGlobal,
        dPrimitiveIds,
        dTransformIds,
        dHitStructs,
        //
        dMatIds,
        dRayIds,
        // I-O
        localData,
        this->globalData,
        rngMem.RNGData(gpu),
        // Constants
        rayCount,
        matData,
        primData,
        dTransforms
    );
}

template<class GlobalData, class LocalData, class RayData,
         class MGroup, class PGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline const char* GPUBoundaryWorkBatch<GlobalData, LocalData, RayData,
                                        MGroup, PGroup, WFunc, SGen>::TypeNameGen(const char* mgOverride,
                                                                                  const char* pgOverride)
{
    const char* pgName = PGroup::TypeName();
    const char* mgName = MGroup::TypeName();
    if(pgOverride) pgName = pgOverride;
    if(mgOverride) mgName = mgOverride;

    static std::string typeName = MangledNames::WorkBatch(pgName,
                                                          mgName);
    return typeName.c_str();
}

template<class GlobalData, class LocalData, class RayData,
         class MGroup, class PGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
GPUBoundaryWorkBatch<GlobalData, LocalData, RayData,
                     MGroup, PGroup, WFunc, SGen>::GPUBoundaryWorkBatch(const GPUMaterialGroupI& mg,
                                                                        const GPUPrimitiveGroupI& pg,
                                                                        const GPUTransformI* const* t)
    : materialGroup(static_cast<const MGroup&>(mg))
    , primitiveGroup(static_cast<const PGroup&>(pg))
    , dTransforms(t)
{}

template<class GlobalData, class LocalData, class RayData, class MGroup, class PGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
void GPUBoundaryWorkBatch<GlobalData, LocalData, RayData,
                          MGroup, PGroup, WFunc, SGen>::Work(// Output
                                                     HitKey* dBoundMatOut,
                                                     RayGMem* dRayOut,
                                                     //  Input
                                                     const RayGMem* dRayIn,
                                                     const PrimitiveId* dPrimitiveIds,
                                                     const TransformId* dTransformIds,
                                                     const HitStructPtr dHitStructs,
                                                     // Ids
                                                     const HitKey* dMatIds,
                                                     const RayId* dRayIds,
                                                     //
                                                     const uint32_t rayCount,
                                                     RNGMemory& rngMem)
{
    // Do Pre-work (initialize local data etc.)
    this->GetReady();

    using PrimitiveGroup = PGroup;
    using PrimitiveData = typename PGroup::PrimitiveData;
    using HitData = typename PGroup::HitData;
    using MaterialData = typename MGroup::Data;
    using MaterialSurface = typename MGroup::Surface;

    // Get Data
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
    const MaterialData matData = MatDataAccessor::Data(materialGroup);

    const uint32_t outRayCount = this->OutRayCount();

    const CudaGPU& gpu = materialGroup.GPU();
    gpu.AsyncGridStrideKC_X
    (
        0,
        rayCount,
        //
        KCBoundaryWork<GlobalData, LocalData, RayData, PGroup,
                       MGroup, WorkF, SurfF>,
        // Args
        // Output
        dBoundMatOut,
        dRayOut,
        this->dAuxOutLocal,
        outRayCount,
        // Input
        dRayIn,
        this->dAuxInGlobal,
        dPrimitiveIds,
        dTransformIds,
        dHitStructs,
        //
        dMatIds,
        dRayIds,
        // I-O
        localData,
        this->globalData,
        rngMem.RNGData(gpu),
        // Constants
        rayCount,
        endpointFinder,
        matData,
        primData,
        dTransforms
    );
}

template<class GlobalData, class LocalData, class RayData, class MGroup, class PGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
TracerError GPUBoundaryWorkBatch<GlobalData, LocalData, RayData,
                         MGroup, PGroup, WFunc, SGen>::GenerateEndpointFinder(const GPUEndpointI* const* sceneEndPoints)
{
    // TODO:
    return TracerError::UNABLE_TO_GENERATE_WORK;
}