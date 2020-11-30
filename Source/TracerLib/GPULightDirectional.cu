#include "GPULightDirectional.cuh"
#include "TypeTraits.h"

SceneError CPULightGroupDirectional::InitializeGroup(const ConstructionDataList& lightNodes,
                                                     const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                     const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                     const MaterialKeyListing& allMaterialKeys,
                                                     double time,
                                                     const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

SceneError CPULightGroupDirectional::ChangeTime(const NodeListing& lightNodes, double time,
                                                const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

TracerError CPULightGroupDirectional::ConstructLights(const CudaSystem&)
{
    // TODO: Implement
    return TracerError::UNABLE_TO_CONSTRUCT_LIGHT;
}

static_assert(IsLightGroupClass<CPULightGroupDirectional>::value,
              "CPULightGroupDirectional is not a Light Group class.");