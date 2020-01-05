#pragma once

#include "GPUEventEstimatorP.h"
#include "DefaultEstimatorsKC.cuh"
#include "TypeTraits.h"


class GPUEventEstimatorBasic final
    : public GPUEventEstimator<BasicEstimatorData,
                               EstimateEventBasic>
{
    public:
        static constexpr const char*    TypeName() { return "BasicEstimator"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                        GPUEventEstimatorBasic() = default;
                                        ~GPUEventEstimatorBasic() = default;

        // Interface
        const char*                     Type() const override;

        SceneError                      Initialize(const NodeListing& lightList,
                                                   // Material Keys
                                                   const MaterialKeyListing& hitKeys,
                                                   const std::map<uint32_t, GPUPrimitiveGroupI>&) override;

        // Constructs Event Estimator
        SceneError                      ConstructEventEstimator(const CudaSystem&) override;
};

static_assert(IsTracerClass<GPUEventEstimatorBasic>::value,
              "GPUEventEstimatorBasic is not a Tracer Class.");