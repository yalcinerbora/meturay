#pragma once

#include "MetaTracerWork.cuh"
#include "TracerKC.cuh"

template<class MGroup, class PGroup,
          SurfaceFunc<MGroup, PGroup> SFunc>
class BasicTracerWork 
    : public MetaTracerBatch<BasicTracerGlobal, EmptyState, RayAuxBasic,
                            MGroup, PGroup, SFunc, BasicWork<MGroup>>
{
       public:
        // Constrcutors & Destructor
                                        BasicTracerWork(const GPUMaterialGroupI&,
                                                        const GPUPrimitiveGroupI&);
                                        ~BasicTracerWork() = default;

        void                            PreWork() override {}
        // We will not bounce more than once
        uint8_t                 OutRayCount() const override { return 0; }
};

template<class MG, class PG,
    SurfaceFunc<MG, PG> SF>
BasicTracerWork<MG, PG, SF>::BasicTracerWork(const GPUMaterialGroupI& mg,
                                             const GPUPrimitiveGroupI& pg)
    : MetaTracerBatch(mg, pg)
{}