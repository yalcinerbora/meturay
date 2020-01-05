#pragma once

#include "GPUEventEstimatorP.h"
#include "DefaultEstimatorsKC.cuh"
#include "TypeTraits.h"

class GPUEventEstimatorEmpty final
    : public GPUEventEstimator<EmptyEstimatorData,
                               EstimateEventEmpty>
{
    public:
        static constexpr const char*    TypeName() { return "EmptyEstimator"; }

    private:
    protected:
    public:  
        // Constructors & Destructor
                                        GPUEventEstimatorEmpty() = default;
                                        ~GPUEventEstimatorEmpty() = default;

        const char*                     Type() const override;
        SceneError                      ConstructEventEstimator(const CudaSystem&) override;
};

static_assert(IsTracerClass<GPUEventEstimatorEmpty>::value,
              "GPUEventEstimatorEmpty is not a Tracer Class.");

inline const char* GPUEventEstimatorEmpty::Type() const
{ 
    return TypeName(); 
}

inline SceneError GPUEventEstimatorEmpty::ConstructEventEstimator(const CudaSystem&)
{
    return  SceneError::OK; 
}