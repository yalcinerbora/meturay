#include "BasicEventEstimator.h"

const char* BasicEventEstimator::Type() const
{
    return TypeName();
}

SceneError BasicEventEstimator::Initialize(const NodeListing& lightList,
                                           // Material Keys
                                           const MaterialKeyListing& hitKeys,
                                           const std::map<uint32_t, GPUPrimitiveGroupI>&)
{
    // TODO:
    return SceneError::OK;
}

SceneError BasicEventEstimator::ConstructEventEstimator(const CudaSystem&)
{
    // TODO:
    return SceneError::OK;
}