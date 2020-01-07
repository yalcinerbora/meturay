#pragma once

#include "GPUEventEstimatorP.h"
#include "DefaultEstimatorsKC.cuh"
#include "TypeTraits.h"

class GPUEventEstimatorEmpty final
    : public GPUEventEstimator<EmptyEstimatorData,
                               EstimateEventEmpty,
                               TerminateEventEmpty>
{
    public:
        static constexpr const char*    TypeName() { return "Empty"; }

    private:
    protected:
    public:  
        // Constructors & Destructor
                                        GPUEventEstimatorEmpty() = default;
                                        ~GPUEventEstimatorEmpty() = default;

        SceneError                      Initialize(const NodeListing& lightList,
                                                   // Material Keys
                                                   const MaterialKeyListing& hitKeys,
                                                   const std::map<uint32_t, const GPUPrimitiveGroupI*>&,
                                                   double time) override;

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

inline SceneError GPUEventEstimatorEmpty::Initialize(const NodeListing& lightList,
                                                     // Material Keys
                                                     const MaterialKeyListing& hitKeys,
                                                     const std::map<uint32_t, const GPUPrimitiveGroupI*>&,
                                                     double time)
{
    // TODO:
    return SceneError::OK;
}