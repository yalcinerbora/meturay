#include "DebugMaterials.cuh"

SceneError BarycentricMat::InitializeGroup(const NodeListing& materialNodes,
                                           const TextureNodeMap&,
                                           const std::map<uint32_t, uint32_t>&,
                                           double, const std::string&)
{
    return GenerateInnerIds(materialNodes);
}

SceneError SphericalMat::InitializeGroup(const NodeListing& materialNodes,
                                         const TextureNodeMap&,
                                         const std::map<uint32_t, uint32_t>&,
                                         double, const std::string&)
{
    return GenerateInnerIds(materialNodes);
}

SceneError SphericalAnisoTestMat::InitializeGroup(const NodeListing& materialNodes,
                                                  const TextureNodeMap&,
                                                  const std::map<uint32_t, uint32_t>&,
                                                  double, const std::string&)
{
    return GenerateInnerIds(materialNodes);
}

SceneError NormalRenderMat::InitializeGroup(const NodeListing& materialNodes,
                                            const TextureNodeMap&,
                                            const std::map<uint32_t, uint32_t>&,
                                            double, const std::string&)
{
    return GenerateInnerIds(materialNodes);
}