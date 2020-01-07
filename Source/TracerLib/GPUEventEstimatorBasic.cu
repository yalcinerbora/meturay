#include "GPUEventEstimatorBasic.h"

const char* GPUEventEstimatorBasic::Type() const
{
    return TypeName();
}

SceneError GPUEventEstimatorBasic::ConstructEventEstimator(const CudaSystem&)
{
    // TODO:
    return SceneError::OK;
}