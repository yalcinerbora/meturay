#pragma once

#include "GPUEventEstimatorP.h"
#include "DefaultEstimatorsKC.cuh"
#include "TypeTraits.h"

class EmptyEventEstimator final
    : public GPUEventEstimator<EmptyEstimatorData,
                               EstimateEventEmpty>
{
    public:
        static constexpr const char*    TypeName() { return "BasicEvent"; }

    private:
    protected:
    public:  
        // Constructors & Destructor
                                        EmptyEventEstimator() = default;
                                        ~EmptyEventEstimator() = default;

        const char*                     Type() const override;
        SceneError                      ConstructEventEstimator(const CudaSystem&) override;
};

static_assert(IsTracerClass<EmptyEventEstimator>::value,
              "EmptyEventEstimator is not a Tracer Class.");

inline const char* EmptyEventEstimator::Type() const
{ 
    return TypeName(); 
}

inline SceneError EmptyEventEstimator::ConstructEventEstimator(const CudaSystem&)
{
    return  SceneError::OK; 
}