#include "GPULightSpot.cuh"
#include "TypeTraits.h"

SceneError CPULightGroupSpot::InitializeGroup(const ConstructionDataList& lightNodes,
                                              const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                              const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                              const MaterialKeyListing& allMaterialKeys,
                                              double time,
                                              const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

SceneError CPULightGroupSpot::ChangeTime(const NodeListing& lightNodes, double time,
                                         const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

TracerError CPULightGroupSpot::ConstructLights(const CudaSystem&)
{
    // TODO: Implement
    return TracerError::UNABLE_TO_CONSTRUCT_LIGHT;
}

static_assert(IsLightGroupClass<CPULightGroupSpot>::value,
              "CPULightGroupSpot is not a Light Group class.");