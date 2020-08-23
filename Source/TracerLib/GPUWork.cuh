#pragma once

#include "GPUWorkI.h"
#include "Random.cuh"
#include "GPUPrimitiveP.cuh"
#include "GPUMaterialP.cuh"
#include "RNGMemory.h"
#include "MangledNames.h"
#include "WorkKernels.cuh"

// Meta Tracer Code
// With custom global Data

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
         WorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc>
class GPUWorkBatch 
    : public GPUWorkBatchD<GlobalData, RayData>
{
    public:
        static const char*              TypeName();
    protected:
        const MGroup&                   materialGroup;
        const PGroup&                   primitiveGroup;

        const GPUTransformI* const*     dTransforms;

        // Per-Bathch Data
        LocalData                       localData;

    public:
        // Constrcutors & Destructor
                                        GPUWorkBatch(const GPUMaterialGroupI&,
                                                     const GPUPrimitiveGroupI&,
                                                     const GPUTransformI* const*);
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

        const GPUPrimitiveGroupI&       PrimitiveGroup() const { return primitiveGroup; }
        const GPUMaterialGroupI&        MaterialGroup() const { return materialGroup; }
      
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

template <class GD, class LD, class RD, class MG, class PG,
          WorkFunc<GD, LD, RD, MG> WF>
inline const char* GPUWorkBatch<GD, LD, RD, MG, PG, WF>::TypeName()
{
    static std::string typeName = MangledNames::WorkBatch(PG::TypeName(),
                                                          MG::TypeName());
    return typeName.c_str();
}

template <class GD, class LD, class RD, class MG, class PG, 
          WorkFunc<GD, LD, RD, MG> WF>
GPUWorkBatch<GD, LD, RD, MG, PG, WF>::GPUWorkBatch(const GPUMaterialGroupI& mg,
                                                   const GPUPrimitiveGroupI& pg,
                                                   const GPUTransformI* const* t)
    : materialGroup(static_cast<const MG&>(mg))
    , primitiveGroup(static_cast<const PG&>(pg))
    , dTransforms(t)
{}

template <class GD, class LD, class RD, class MG, class PG, 
          WorkFunc<GD, LD, RD, MG> WF>
void GPUWorkBatch<GD, LD, RD, MG, PG, WF>::Work(// Output
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
    GetReady();

    using PrimitiveGroup = typename PG;
    using PrimitiveData = typename PG::PrimitiveData;
    using HitData = typename PG::HitData;
    using MaterialData = typename MG::Data;
    using MaterialSurface = typename MG::Surface;
    // Fetch surface function from primitive list
    //using SFuncType = SurfaceFunc<MaterialSurface, HitData, PrimitiveData>;
    constexpr auto SFunc = PG::GetSurfaceFunction<typename MG::Surface>();

    // Get Data
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
    const MaterialData matData = MatDataAccessor::Data(materialGroup);    

    const uint32_t outRayCount = OutRayCount();

    const CudaGPU& gpu = materialGroup.GPU();
    gpu.AsyncGridStrideKC_X
    (
        0,
        rayCount,
        //
        KCWork<GD, LD, RD, PG, MG, WF, SFunc>,
        // Args
        // Output
        dBoundMatOut,
        dRayOut,
        dAuxOutLocal,
        outRayCount,
        // Input
        dRayIn,
        dAuxInGlobal,
        dPrimitiveIds,
        dTransformIds,
        dHitStructs,
        //
        dMatIds,
        dRayIds,
        // I-O 
        localData,
        globalData,
        rngMem.RNGData(gpu),
        // Constants
        rayCount,
        matData,
        primData,
        dTransforms
    );
}