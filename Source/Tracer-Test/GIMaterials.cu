#include "GIMaterials.cuh"
#include "MaterialNodeReaders.h"

BasicPathTraceMat::BasicPathTraceMat(const CudaGPU& gpu)
    : GPUMaterialGroup(gpu)
{}

SceneError BasicPathTraceMat::InitializeGroup(const NodeListing& materialNodes, double time,
                                              const std::string& scenePath)
{
    constexpr const char* ALBEDO = "albedo";

    std::vector<Vector3> albedoCPU;
    uint32_t i = 0;
    for(const auto& sceneNode : materialNodes)
    {
        std::vector<Vector3> albedos = sceneNode->AccessVector3(ALBEDO);
        albedoCPU.insert(albedoCPU.end(), albedos.begin(), albedos.end());

        const auto& ids = sceneNode->Ids();
        for(IdPair id : ids)
        {
            innerIds.emplace(std::make_pair(id.first, i));
            i++;
        }
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

LightBoundaryMat::LightBoundaryMat(const CudaGPU& gpu)
    : GPUMaterialGroup(gpu)
{}

SceneError LightBoundaryMat::InitializeGroup(const NodeListing& materialNodes, double time,
                                             const std::string& scenePath)
{
    constexpr const char* IRRADIANCE = "irradiance";

    std::vector<Vector3> irradianceCPU;
    uint32_t i = 0;
    for(const auto& sceneNode : materialNodes)
    {
        std::vector<Vector3> irradiances = sceneNode->AccessVector3(IRRADIANCE);
        irradianceCPU.insert(irradianceCPU.end(), irradiances.begin(), irradiances.end());

        const auto& ids = sceneNode->Ids();
        for(IdPair id : ids)
        {
            innerIds.emplace(std::make_pair(id.first, i));
            i++;
        }
    }

    // Alloc etc
    size_t dIrradianceSize = irradianceCPU.size() * sizeof(Vector3);
    memory = std::move(DeviceMemory(dIrradianceSize));
    Vector3f* dIrradiance = static_cast<Vector3f*>(memory);
    CUDA_CHECK(cudaMemcpy(dIrradiance, irradianceCPU.data(), dIrradianceSize,
               cudaMemcpyHostToDevice));

    dData = ConstantIrradianceMatData{dIrradiance};
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

template class GPUMaterialBatch<TracerBasic,
                                LightBoundaryMat,
                                GPUPrimitiveEmpty,
                                EmptySurfaceFromEmpty>;

template class GPUMaterialBatch<TracerBasic,
                                LightBoundaryMat,
                                GPUPrimitiveTriangle,
                                EmptySurfaceFromTri>;

template class GPUMaterialBatch<TracerBasic,
                                LightBoundaryMat,
                                GPUPrimitiveSphere,
                                EmptySurfaceFromSphr>;