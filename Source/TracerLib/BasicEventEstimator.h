#pragma once

#include "GPUEventEstimatorP.h"
#include "DefaultEstimatorsKC.cuh"
#include "TypeTraits.h"


class BasicEventEstimator final
    : public GPUEventEstimator<BasicEstimatorData,
                               EstimateEventBasic>
{
    public:
        static constexpr const char*    TypeName() { return "BasicEvent"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                        BasicEventEstimator() = default;
                                        ~BasicEventEstimator() = default;

        // Interface
        const char*                     Type() const override;

        SceneError                      Initialize(const NodeListing& lightList,
                                                   // Material Keys
                                                   const MaterialKeyListing& hitKeys,
                                                   const std::map<uint32_t, GPUPrimitiveGroupI>&) override;

        // Constructs Event Estimator
        SceneError                      ConstructEventEstimator(const CudaSystem&) override;
};

static_assert(IsTracerClass<BasicEventEstimator>::value,
              "BasicEventEstimator is not a Tracer Class.");