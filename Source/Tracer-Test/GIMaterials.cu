#include "GIMaterials.cuh"
#include "MaterialNodeReaders.h"

BasicPathTraceMat::BasicPathTraceMat(int gpuId)
    : GPUMaterialGroup(gpuId)
{}

SceneError BasicPathTraceMat::InitializeGroup(const NodeListing& materialNodes, double time,
                                              const std::string& scenePath)
{
    constexpr const char* ALBEDO = "albedo";

    std::vector<Vector3> albedoCPU;
    for(const auto& sceneNode : materialNodes)
    {
        std::vector<Vector3> albedos = sceneNode->AccessVector3(ALBEDO);
        albedoCPU.insert(albedoCPU.end(), albedos.begin(), albedos.end());
    }

    // Alloc etc
    size_t dAlbedoSize = albedoCPU.size() * sizeof(Vector3);
    memory = std::move(DeviceMemory(dAlbedoSize));
    Vector3f* dAlbedo = static_cast<Vector3f*>(memory);
    CUDA_CHECK(cudaMemcpy(dAlbedo, albedoCPU.data(), dAlbedoSize,
                          cudaMemcpyHostToDevice));

    dData = ConstantAlbedoMatData{dAlbedo};
    return SceneError::OK;
}

SceneError BasicPathTraceMat::ChangeTime(const NodeListing& materialNodes, double time,
                                         const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::NO_LOGIC_FOR_MATERIAL;
}

int BasicPathTraceMat::InnerId(uint32_t materialId) const
{
    return innerIds.at(materialId);
}

LightBoundaryMat::LightBoundaryMat(int gpuId)
    : GPUMaterialGroup(gpuId)
{}

SceneError LightBoundaryMat::InitializeGroup(const NodeListing& materialNodes, double time,
                                             const std::string& scenePath)
{
    constexpr const char* ALBEDO = "albedo";

    std::vector<Vector3> albedoCPU;
    for(const auto& sceneNode : materialNodes)
    {
        std::vector<Vector3> albedos = sceneNode->AccessVector3(ALBEDO);
        albedoCPU.insert(albedoCPU.end(), albedos.begin(), albedos.end());
    }

    // Alloc etc
    size_t dAlbedoSize = albedoCPU.size() * sizeof(Vector3);
    memory = std::move(DeviceMemory(dAlbedoSize));
    Vector3f* dAlbedo = static_cast<Vector3f*>(memory);
    CUDA_CHECK(cudaMemcpy(dAlbedo, albedoCPU.data(), dAlbedoSize,
               cudaMemcpyHostToDevice));

    dData = ConstantAlbedoMatData{dAlbedo};
    return SceneError::OK;
}

SceneError LightBoundaryMat::ChangeTime(const NodeListing& materialNodes, double time,
                                        const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::NO_LOGIC_FOR_MATERIAL;
}

int LightBoundaryMat::InnerId(uint32_t materialId) const
{
    return innerIds.at(materialId);
}

// Material Batch Implementations
template class GPUMaterialBatch<TracerBasic,
                                BasicPathTraceMat,
                                GPUPrimitiveTriangle,
                                BasicSurfaceFromTri>;

template class GPUMaterialBatch<TracerBasic,
                                BasicPathTraceMat,
                                GPUPrimitiveSphere,
                                BasicSurfaceFromSphr>;

template class GPUMaterialBatch<TracerBasic,
                                LightBoundaryMat,
                                GPUPrimitiveEmpty,
                                EmptySurfaceFromEmpty>;