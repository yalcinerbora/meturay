#pragma once

#include "TracerLib/GPUWorkI.h"
#include "TracerLib/Random.cuh"
#include "TracerLib/GPUPrimitiveP.cuh"
#include "TracerLib/GPUMaterialP.cuh"
#include "TracerLib/RNGMemory.h"
#include "TracerLib/MangledNames.h"
#include "TracerLib/WorkKernels.cuh"

// Meta Tracer Code
// With custom global Data

// Material/Primitive invaritant part of the code
template<class GlobalData, class RayData>
class MetaWorkBatchData : public GPUWorkBatchI
{
    protected:
        // Ray Auxiliary Input and output pointers
        // which are global (not local)
        const RayData*      dRayDataIn = nullptr;
        RayData*            dRayDataOut = nullptr;

        // GPU Friendly Struct which will be directly passed to the kernel call
        GlobalData          globalData;
        

    public:
            // Constructors & Destructor
                            MetaWorkBatchData() = default;
                            ~MetaWorkBatchData() = default;

            void            SetGlobalData(const GlobalData&);
            void            SetRayDataPtrs(RayData* rayDataOut,
                                          const RayData* rayDataIn);

};

template<class GlobalData, class LocalData, class RayData,
         class MGroup, class PGroup,
         SurfaceFunc<MGroup, PGroup> SFunc,
         WorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc>
class MetaTracerBatch 
    : public MetaWorkBatchData<GlobalData, RayData>
{
    public:
        static const char*              TypeName();

    private:        
        static constexpr auto           GenerateSurface = SFunc;

        // Per-Bathch Data
        LocalData                       localData;

    protected:
        const MGroup&                   materialGroup;
        const PGroup&                   primitiveGroup;

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
      
};

template<class GD, class RD>
inline void MetaWorkBatchData<GD, RD>::SetGlobalData(const GD& d)
{
    globalData = d;
}

template<class GD, class RD>
void MetaWorkBatchData<GD, RD>::SetRayDataPtrs(RD* dRDOut,
                                               const RD* dRDIn)
{
    dRayDataIn = dRDIn;
    dRayDataOut = dRDOut;
}

template <class GD, class LD, class RD, class MG, class PG,
          SurfaceFunc<MG, PG> SF, WorkFunc<GD, LD, RD, MG> WF>
inline const char* MetaTracerBatch<GD, LD, RD, MG, PG, SF, WF>::TypeName()
{
    static std::string typeName = MangledNames::WorkBatch(PG::TypeName(),
                                                          MG::TypeName());
    return typeName.c_str();
}

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
    GetReady();

    using PrimitiveData = typename PG::PrimitiveData;
    using MaterialData = typename MG::Data;
    
    // Get Data
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
    const MaterialData matData = MatDataAccessor::Data(materialGroup);    

    const uint32_t outRayCount = OutRayCount();
    RD* dAuxOutLocal = dRayDataOut + outputOffset;

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
        dRayDataIn,
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