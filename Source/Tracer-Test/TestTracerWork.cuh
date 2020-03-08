#pragma once


#include "TracerLib/GPUWorkI.h"

//template<class GlobalState, class LocalState,
//         class RayAuxiliary, class PGroup, class MGroup,
//         WorkFunc<GlobalState, LocalState, RayAuxiliary, MGroup> WFunc, 
//         SurfaceFunc<MGroup, PGroup> SurfFunc>

template <class MGroup, class PGroup>
class TestPathTracerBatch final : public GPUWorkBatchI
{
    private:
        const MGroup&                   materialGroup;
        const PGroup&                   primitiveGroup;

    public:
        // Constrcutors & Destructor
                                        TestPathTracerBatch(const GPUMaterialGroupI&,
                                                      const GPUPrimitiveGroupI&);
                                        ~TestPathTracerBatch() = default;

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

};

template <class MG, class PG>
TestPathTracerBatch<MG, PG>::TestPathTracerBatch(EventEstimatorI
                                                 //
                                                 const GPUMaterialGroupI& mg,
                                                 const GPUPrimitiveGroupI& pg)
    : materialGroup(static_cast<const MGroup&>(mg))
    , primitiveGroup(static_cast<const PGroup&>(pg))
{}

