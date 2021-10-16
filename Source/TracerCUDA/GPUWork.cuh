#pragma once

#include "GPUWorkI.h"
#include "Random.cuh"
#include "GPUPrimitiveP.cuh"
#include "GPUMaterialP.cuh"
#include "RNGMemory.h"
#include "MangledNames.h"
#include "WorkKernels.cuh"
#include "GPULightNull.cuh"

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

template<class GlobalData, class RayData>
class GPUWorkBatchIntermediateI : public GPUWorkBatchD<GlobalData, RayData>
{
    public:
        virtual                             ~GPUWorkBatchIntermediateI() = default;
        // Every MaterialBatch is available for a specific primitive / material data
        virtual const GPUPrimitiveGroupI&   PrimitiveGroup() const = 0;
        virtual const GPUMaterialGroupI&    MaterialGroup() const = 0;
};

template<class GlobalData, class RayData>
class GPUWorkBatchBoundaryI : public GPUWorkBatchD<GlobalData, RayData>
{
    public:
        virtual                             ~GPUWorkBatchBoundaryI() = default;
        // Every MaterialBatch is available for a specific primitive / material data
        virtual const CPUEndpointGroupI&    EndpointGroup() const = 0;
};

template<class GlobalData, class LocalData, class RayData,
         class MGroup, class PGroup,
         WorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
class GPUWorkBatch
    : public GPUWorkBatchIntermediateI<GlobalData, RayData>
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

template<class GlobalData, class LocalData,
         class RayData, class EGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, EGroup> WFunc>
class GPUBoundaryWorkBatch
    : public GPUWorkBatchBoundaryI<GlobalData, RayData>
{
    public:
        static const char*              TypeNameGen(const char* nameOverride = nullptr);
        static const char*              TypeName() {return TypeNameGen();}

    private:
        using SF =                      SurfaceFunc<typename EGroup::Surface,
                                                    typename EGroup::HitData,
                                                    typename EGroup::PrimitiveData>;
        using WF =                      BoundaryWorkFunc<GlobalData, LocalData,
                                                         RayData, EGroup>;

        static constexpr SF             SurfF = EGroup::SurfF;
        static constexpr WF             WorkF = WFunc;

    protected:
        const EGroup&                   endpointGroup;

        const GPUTransformI* const*     dTransforms;

        // Per-Bathch Data
        LocalData                       localData;

    public:
        // Constrcutors & Destructor
                                        GPUBoundaryWorkBatch(const CPUEndpointGroupI& eg,
                                                             const GPUTransformI* const* t);
                                        ~GPUBoundaryWorkBatch() = default;

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

        uint8_t                         OutRayCount() const override { return 0; };
        const CPUEndpointGroupI&        EndpointGroup() const override { return endpointGroup; }
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

// ======================================================= //
//                      BOUNDARY WORK                      //
// ======================================================= //

template<class GlobalData, class LocalData,
         class RayData, class EGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, EGroup> WFunc>
inline const char* GPUBoundaryWorkBatch<GlobalData, LocalData, RayData,
                                        EGroup, WFunc>::TypeNameGen(const char* nameOverride)
{
    const char* name = EGroup::TypeName();
    if(nameOverride) name = nameOverride;

    static std::string typeName = MangledNames::BoundaryWorkBatch(name);
    return typeName.c_str();
}

template<class GlobalData, class LocalData,
         class RayData, class EGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, EGroup> WFunc>
GPUBoundaryWorkBatch<GlobalData, LocalData, RayData,
                     EGroup, WFunc>::GPUBoundaryWorkBatch(const CPUEndpointGroupI& eg,
                                                          const GPUTransformI* const* t)
    : endpointGroup(static_cast<const EGroup&>(eg))
    , dTransforms(t)
{}

template<class GlobalData, class LocalData,
         class RayData, class EGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, EGroup> WFunc>
void GPUBoundaryWorkBatch<GlobalData, LocalData, RayData,
                          EGroup, WFunc>::Work(// Output
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
    // Special Case Skip Null Light
    if constexpr(std::is_same_v<EGroup, CPULightGroupNull>)
        return;

    // Do Pre-work (initialize local data etc.)
    this->GetReady();

    using PrimitiveGroup    = typename EGroup::PrimitiveGroup;
    using PrimitiveData     = typename EGroup::PrimitiveData;
    using HitData           = typename EGroup::HitData;
    using Surface           = typename EGroup::Surface;
    using GPUEndpointType   = typename EGroup::GPUType;

    // Get Data
    PrimitiveData primData;
    if constexpr(std::is_same_v<EGroup, CPULightGroupNull>)
        primData = {};
    else
        primData = PrimDataAccessor::Data(endpointGroup.PrimGroup());

    const GPUEndpointType* dEndpoints = endpointGroup.GPULightsDerived();

    const uint32_t outRayCount = this->OutRayCount();
    const CudaGPU& gpu = endpointGroup.GPU();
    gpu.AsyncGridStrideKC_X
    (
        0,
        rayCount,
        //
        KCBoundaryWork<GlobalData, LocalData, RayData,
                       EGroup, WorkF, SurfF>,
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
        dEndpoints,
        primData,
        dTransforms
    );
}
