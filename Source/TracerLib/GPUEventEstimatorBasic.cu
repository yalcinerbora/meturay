#include "GPUEventEstimatorBasic.h"

const char* GPUEventEstimatorBasic::Type() const
{
    return TypeName();
}

SceneError GPUEventEstimatorBasic::Initialize(const NodeListing& lightList,
                                           // Material Keys
                                              const MaterialKeyListing& hitKeys,
                                              const std::map<uint32_t, GPUPrimitiveGroupI>&)
{
    // TODO:
    return SceneError::OK;
}

SceneError GPUEventEstimatorBasic::ConstructEventEstimator(const CudaSystem&)
{
    // TODO:
    return SceneError::OK;
}