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
                                                   const MaterialKeyListing& materialKeys,
                                                   const std::vector<const GPUPrimitiveGroupI*>&,
                                                   double time) override;

        const char*                     Type() const override;
        TracerError                     Construct(const CudaSystem&) override;
};

static_assert(IsTracerClass<GPUEventEstimatorEmpty>::value,
              "GPUEventEstimatorEmpty is not a Tracer Class.");

inline const char* GPUEventEstimatorEmpty::Type() const
{ 
    return TypeName(); 
}

inline TracerError GPUEventEstimatorEmpty::Construct(const CudaSystem&)
{
    return TracerError::OK;
}

inline SceneError GPUEventEstimatorEmpty::Initialize(const NodeListing&,
                                                     // Material Keys
                                                     const MaterialKeyListing&,
                                                     const std::vector<const GPUPrimitiveGroupI*>&,
                                                     double time)
{
    // TODO:
    return SceneError::OK;
}