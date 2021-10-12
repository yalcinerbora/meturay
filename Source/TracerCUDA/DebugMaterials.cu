#include "DebugMaterials.cuh"

SceneError BarycentricMat::InitializeGroup(const NodeListing& materialNodes,
                                           const TextureNodeMap& textures,
                                           const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                           double time, const std::string& scenePath)
{
    return GenerateInnerIds(materialNodes);
}

SceneError SphericalMat::InitializeGroup(const NodeListing& materialNodes,
                                         const TextureNodeMap& textures,
                                         const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                         double time, const std::string& scenePath)
{
    return GenerateInnerIds(materialNodes);
}

SceneError NormalRenderMat::InitializeGroup(const NodeListing& materialNodes,
                                            const TextureNodeMap& textures,
                                            const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                            double time, const std::string& scenePath)
{
    return GenerateInnerIds(materialNodes);
}