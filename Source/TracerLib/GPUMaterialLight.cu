#include "GPUMaterialLight.cuh"
#include "RayLib/ColorConversion.h"

SceneError LightMatConstant::InitializeGroup(const NodeListing& materialNodes, double time,
                                             const std::string& scenePath)
{
    constexpr const char* RADIANCE = "radiance";

    std::vector<Vector3> radianceCPU;
    uint32_t i = 0;
    for(const auto& sceneNode : materialNodes)
    {
        std::vector<Vector3> radiances = sceneNode->AccessVector3(RADIANCE);
        radianceCPU.insert(radianceCPU.end(), radiances.begin(), radiances.end());
    
        // Calculate Distributions
        for(const Vector3& r : radiances)
        {
            float lum = Utility::RGBToLuminance(r);
            lightRadianceDistributions.emplace_back(std::vector<float>(1, lum));
        }
        // Generate Id pairs
        const auto& ids = sceneNode->Ids();
        for(IdPair id : ids)
        {
            innerIds.emplace(std::make_pair(id.first, i));
            i++;
        }
    }

    // Alloc etc
    size_t dRadianceSize = radianceCPU.size() * sizeof(Vector3);
    memory = std::move(DeviceMemory(dRadianceSize));
    Vector3f* dRadiances = static_cast<Vector3f*>(memory);
    CUDA_CHECK(cudaMemcpy(dRadiances, radianceCPU.data(), dRadianceSize,
                          cudaMemcpyHostToDevice));

    dData = LightMatData{dRadiances};
    return SceneError::OK;
}

SceneError LightMatConstant::ChangeTime(const NodeListing& materialNodes, double time,
                                        const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

const GPUDistribution1D& LightMatConstant::LightDistribution(uint32_t materialId) const
{
    return lightRadianceDistributions[innerIds.at(materialId)].DistributionGPU();
}

SceneError LightMatTextured::InitializeGroup(const NodeListing& materialNodes, double time,
                                             const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

SceneError LightMatTextured::ChangeTime(const NodeListing& materialNodes, double time,
                                        const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

const GPUDistribution2D& LightMatTextured::LightDistribution(uint32_t materialId) const
{
    return lightRadianceDistributions[innerIds.at(materialId)].DistributionGPU();
}

SceneError LightMatCube::InitializeGroup(const NodeListing& materialNodes, double time,
                                         const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

SceneError LightMatCube::ChangeTime(const NodeListing& materialNodes, double time,
                                   const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

const GPUDistribution2D& LightMatCube::LightDistribution(uint32_t materialId) const
{
    return lightRadianceDistributions[innerIds.at(materialId)].DistributionGPU();
}