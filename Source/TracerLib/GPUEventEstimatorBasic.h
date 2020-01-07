#pragma once

#include "GPUEventEstimatorP.h"
#include "DefaultEstimatorsKC.cuh"
#include "TypeTraits.h"


class GPUEventEstimatorBasic final
    : public GPUEventEstimator<BasicEstimatorData,
                               EstimateEventBasic,
                               TerminateEventBasic>
{
    public:
        static constexpr const char*    TypeName() { return "Basic"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                        GPUEventEstimatorBasic() = default;
                                        ~GPUEventEstimatorBasic() = default;

        // Interface
        const char*                     Type() const override;

        // Constructs Event Estimator
        SceneError                      ConstructEventEstimator(const CudaSystem&) override;
};

static_assert(IsTracerClass<GPUEventEstimatorBasic>::value,
              "GPUEventEstimatorBasic is not a Tracer Class.");