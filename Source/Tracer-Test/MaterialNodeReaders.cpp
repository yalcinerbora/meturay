#include "MaterialNodeReaders.h"
#include "MaterialDataStructs.h"

#include "RayLib/SceneIO.h"

#include "TracerLib/DeviceMemory.h"
#include "TracerLib/SceneFileNode.h"

ConstantAlbedoMatData ConstantAlbedoMatRead(DeviceMemory& mem,
                                            const std::set<SceneFileNode>& materialNodes,
                                            double time)
{
    constexpr const char* ALBEDO = "albedo";

    std::vector<Vector3> albedoCPU;
    albedoCPU.reserve(materialNodes.size());

    for(const auto& sceneNode : materialNodes)
    {
        const nlohmann::json& node = sceneNode;
        albedoCPU.push_back(SceneIO::LoadVector<3, float>(node[ALBEDO], time));
    }

    // Alloc etc
    size_t dAlbedoSize = albedoCPU.size() * sizeof(Vector3);
    mem = std::move(DeviceMemory(dAlbedoSize));
    Vector3f* dAlbedo = static_cast<Vector3f*>(mem);
    CUDA_CHECK(cudaMemcpy(dAlbedo, albedoCPU.data(), dAlbedoSize,
                          cudaMemcpyHostToDevice));

    return ConstantAlbedoMatData
    {
        dAlbedo
    };
}

ConstantBoundaryMatData ConstantBoundaryMatRead(const std::set<SceneFileNode>& materialNodes,
                                                double time)
{
    constexpr const char* ALBEDO = "albedo";
    if(materialNodes.size() == 0) return {};

    const SceneFileNode& sceneNode = *materialNodes.begin();
    const nlohmann::json& node = sceneNode;
    Vector3 background = SceneIO::LoadVector<3, float>(node[ALBEDO], time);
    return {background};
}