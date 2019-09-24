#include "BasicMaterials.cuh"
#include "MaterialNodeReaders.h"

BasicMat::BasicMat(int gpuId)
    : GPUMaterialGroup(gpuId)
{}

SceneError BasicMat::InitializeGroup(const NodeListing& materialNodes, double time)
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

SceneError BasicMat::ChangeTime(const NodeListing& materialNodes, double time)
{
    // TODO: Implement
    return SceneError::NO_LOGIC_FOR_MATERIAL;
}

int BasicMat::InnerId(uint32_t materialId) const
{
    return innerIds.at(materialId);
}

BarycentricMat::BarycentricMat(int gpuId)
    : GPUMaterialGroup(gpuId)
{}

SceneError BarycentricMat::InitializeGroup(const NodeListing& materialNodes, double time)
{
    // Nothing to initialize
    return SceneError::OK;
}

SceneError BarycentricMat::ChangeTime(const NodeListing& materialNodes, double time)
{
    // Nothing to change
    return SceneError::OK;
}

int BarycentricMat::InnerId(uint32_t materialId) const
{
    // Inner id is irrelevant since there is not data for this material
    return 0;
}

SphericalMat::SphericalMat(int gpuId)
    : GPUMaterialGroup(gpuId)
{}

SceneError SphericalMat::InitializeGroup(const NodeListing& materialNodes, double time)
{
    // Nothing to initialize
    return SceneError::OK;
}

SceneError SphericalMat::ChangeTime(const NodeListing& materialNodes, double time)
{
    // Nothing to change
    return SceneError::OK;
}

int SphericalMat::InnerId(uint32_t materialId) const
{
    // Inner id is irrelevant since there is not data for this material
    return 0;
}

// Material Batches
template class GPUMaterialBatch<TracerBasic,
                                BasicMat,
                                GPUPrimitiveEmpty,
                                EmptySurfaceFromEmpty>;

template class GPUMaterialBatch<TracerBasic,
                                BasicMat,
                                GPUPrimitiveTriangle,
                                EmptySurfaceFromTri>;

template class GPUMaterialBatch<TracerBasic,
                                BasicMat,
                                GPUPrimitiveSphere,
                                EmptySurfaceFromSphr>;

template class GPUMaterialBatch<TracerBasic,
                                BarycentricMat,
                                GPUPrimitiveTriangle,
                                BarySurfaceFromTri>;

template class GPUMaterialBatch<TracerBasic,
                                SphericalMat,
                                GPUPrimitiveSphere,
                                SphrSurfaceFromSphr>;