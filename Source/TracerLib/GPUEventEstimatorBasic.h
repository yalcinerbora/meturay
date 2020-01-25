#pragma once

#include "GPUEventEstimatorP.h"
#include "DefaultEstimatorsKC.cuh"
#include "TypeTraits.h"
#include "TracerLib/DeviceMemory.h"

class GPUEventEstimatorBasic final
    : public GPUEventEstimator<BasicEstimatorData,
                               EstimateEventBasic,
                               TerminateEventBasic>
{
    public:
        static constexpr const char*    TypeName() { return "Basic"; }

    private:
        DeviceMemory                    memory;

    protected:
    public:
        // Constructors & Destructor
                                        GPUEventEstimatorBasic() = default;
                                        ~GPUEventEstimatorBasic() = default;

        // Interface
        const char*                     Type() const override;

        // Constructs Event Estimator
        TracerError                     Construct(const CudaSystem&) override;
};

static_assert(IsTracerClass<GPUEventEstimatorBasic>::value,
              "GPUEventEstimatorBasic is not a Tracer Class.");