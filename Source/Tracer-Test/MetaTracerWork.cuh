#pragma once

#include "TracerLib/GPUWorkI.h"
#include "TracerLib/Random.cuh"
#include "TracerLib/GPUPrimitiveP.cuh"
#include "TracerLib/GPUMaterialP.cuh"
#include "TracerLib/RNGMemory.h"

#include "TracerLib/WorkKernels.cuh"

// Meta Tracer Code
// With custom global Data

template<class GlobalData, class LocalData, class RayData,
         class MGroup, class PGroup,
         SurfaceFunc<MGroup, PGroup> SFunc,
         WorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc>
class MetaTracerBatch : public GPUWorkBatchI
{
    private:
        const MGroup&                   materialGroup;
        const PGroup&                   primitiveGroup;

        static constexpr auto           GenerateSurface = SFunc;

        // Global Data
        GlobalData                      globalData;
        // Per-Bathch Data
        LocalData                       localData;
        // Per-Ray Data
        const RayData*                  dRayAuxIn;
        RayData*                        dRayAuxOut;

    public:
        // Constrcutors & Destructor
                                        MetaTracerBatch(const GPUMaterialGroupI&,
                                                        const GPUPrimitiveGroupI&);
                                        ~MetaTracerBatch() = default;

        void                            Work(// Output
                                             HitKey* dBoundMatOut,
                                             RayGMem* dRayOut,
                                             //  Input
                                             const RayGMem* dRayIn,
                                             const PrimitiveId* dPrimitiveIds,
                                             const HitStructPtr dHitStructs,
                                             // Ids
                                             const HitKey* dMatIds,
                                             const RayId* dRayIds,
                                             // 
                                             const uint32_t outputOffset,
                                             const uint32_t rayCount,
                                             RNGMemory& rngMem) override;

        const GPUPrimitiveGroupI&       PrimitiveGroup() const { return primitiveGroup; }
        const GPUMaterialGroupI&        MaterialGroup() const { return materialGroup; }

        void                            SetGlobalData(const GlobalData&);
        
        virtual void                    PreWork() = 0;
        ////// We will not bounce more than once
        //uint8_t                 OutRayCount() const { return 0; }

};

template <class GD, class LD, class RD, class MG, class PG, 
          SurfaceFunc<MG, PG> SF, WorkFunc<GD, LD, RD, MG> WF>
MetaTracerBatch<GD, LD, RD, MG, PG, SF, WF>::MetaTracerBatch(const GPUMaterialGroupI& mg,
                                                         const GPUPrimitiveGroupI& pg)
    : materialGroup(static_cast<const MG&>(mg))
    , primitiveGroup(static_cast<const PG&>(pg))
{}

template <class GD, class LD, class RD, class MG, class PG, 
          SurfaceFunc<MG, PG> SF, WorkFunc<GD, LD, RD, MG> WF>
void MetaTracerBatch<GD, LD, RD, MG, PG, SF, WF>::Work(// Output
                                                       HitKey* dBoundMatOut,
                                                       RayGMem* dRayOut,
                                                       //  Input
                                                       const RayGMem* dRayIn,
                                                       const PrimitiveId* dPrimitiveIds,
                                                       const HitStructPtr dHitStructs,
                                                       // Ids
                                                       const HitKey* dMatIds,
                                                       const RayId* dRayIds,
                                                       // 
                                                       const uint32_t outputOffset,
                                                       const uint32_t rayCount,
                                                       RNGMemory& rngMem)
{
    // Do Pre-work (initialize local data etc.)
    PreWork();

    using PrimitiveData = typename PG::PrimitiveData;
    using MaterialData = typename MG::Data;
    
    // Get Data
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
    const MaterialData matData = MatDataAccessor::Data(materialGroup);    

    const uint32_t outRayCount = OutRayCount();
    RD* dAuxOutLocal = dRayAuxOut + outputOffset;

    const CudaGPU& gpu = materialGroup.GPU();

    gpu.AsyncGridStrideKC_X
    (
        0,
        rayCount,
        //
        KCWork<GD, LD, RD, PG, MG, 
               WF, SF>,
        // Args
        // Output
        dBoundMatOut,
        dRayOut,
        dAuxOutLocal,
        outRayCount,
        // Input
        dRayIn,
        dRayAuxIn,
        dPrimitiveIds,
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
        primData
    );
}

template <class GD, class LD, class RD, class MG, class PG, 
          SurfaceFunc<MG, PG> SF, WorkFunc<GD, LD, RD, MG> WF>
void MetaTracerBatch<GD, LD, RD, MG, PG, SF, WF>::SetGlobalData(const GD& d)
{
    globalData = d;
}