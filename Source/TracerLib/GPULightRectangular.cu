#include "GPULightRectangular.cuh"
#include "TypeTraits.h"

SceneError CPULightGroupRectangular::InitializeGroup(const ConstructionDataList& lightNodes,
                                               const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                               const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                               const MaterialKeyListing& allMaterialKeys,
                                               double time,
                                               const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

SceneError CPULightGroupRectangular::ChangeTime(const NodeListing& lightNodes, double time,
                                          const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

TracerError CPULightGroupRectangular::ConstructLights(const CudaSystem&)
{
    // TODO: Implement
    return TracerError::UNABLE_TO_CONSTRUCT_LIGHT;
}

static_assert(IsLightGroupClass<CPULightGroupRectangular>::value,
              "CPULightGroupRectangular is not a Light Group class.");