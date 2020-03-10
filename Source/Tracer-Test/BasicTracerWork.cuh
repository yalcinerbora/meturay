#pragma once

#include "TracerLib/GPUWorkI.h"
#include "TracerLib/Random.cuh"
#include "TracerLib/GPUPrimitiveP.cuh"
#include "TracerLib/GPUMaterialP.cuh"

#include "TracerLib/WorkKernels.cuh"
#include "BasicTracerKC.cuh"

template <class MGroup, class PGroup,
          SurfaceFunc<MGroup, PGroup> SFunc>
class BasicTracerBatch final : public GPUWorkBatchI
{
    private:
        const MGroup&                   materialGroup;
        const PGroup&                   primitiveGroup;

        static constexpr auto           GenerateSurface = SFunc;

        // Per Iteration Data
        BasicTracerGlobal               globalData;
        BasicTracerLocal                localData;
        // Auxiliary Pointers
        const RayAuxBasic*              dAuxIn;
        RayAuxBasic*                    dAuxOut;

    public:
        // Constrcutors & Destructor
                                        BasicTracerBatch(const GPUMaterialGroupI&,
                                                         const GPUPrimitiveGroupI&);
                                        ~BasicTracerBatch() = default;

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
                                             RNGMemory& rngMem) const;

        const GPUPrimitiveGroupI&       PrimitiveGroup() const { return primitiveGroup; }
        const GPUMaterialGroupI&        MaterialGroup() const { return materialGroup; }

        // We will not bounce more than once
        virtual uint8_t                 OutRayCount() const { return 0; }

};

template <class MG, class PG, SurfaceFunc<MG, PG> SF>
BasicTracerBatch<MG, PG, SF>::BasicTracerBatch(const GPUMaterialGroupI& mg,
                                           const GPUPrimitiveGroupI& pg)
    : materialGroup(static_cast<const MG&>(mg))
    , primitiveGroup(static_cast<const PG&>(pg))
{}

template <class MG, class PG, SurfaceFunc<MG, PG> SF>
void BasicTracerBatch<MG, PG, SF>::Work(// Output
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
                                        RNGMemory& rngMem) const
{
    using PrimitiveData = typename PG::PrimitiveData;
    using MaterialData = typename MG::Data;
    
    // Get Data
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
    const MaterialData matData = MatDataAccessor::Data(materialGroup);    

    const uint32_t outRayCount = OutRayCount();
    RayAuxBasic* dAuxOutLocal = dAuxOut + outputOffset;

    const CudaGPU& gpu = materialGroup.GPU();

    gpu.AsyncGridStrideKC_X
    (
        0,
        rayCount,
        //
        KCWork<BasicTracerGlobal, BasicTracerLocal, 
               RayAuxBasic, PG, MG, BasicWork<MG>, SF>,
        // Args
        // Output
        dBoundMatOut,
        dRayOut,
        dAuxOutLocal,
        outRayCount,
        // Input
        dRayIn,
        dAuxIn,
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