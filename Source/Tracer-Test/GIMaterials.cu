#include "GIMaterials.cuh"
#include "MaterialNodeReaders.h"

GIAlbedoMat::GIAlbedoMat(int gpuId)
    : GPUMaterialGroup(gpuId)
{}

SceneError GIAlbedoMat::InitializeGroup(const NodeListing& materialNodes, double time,
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

SceneError GIAlbedoMat::ChangeTime(const NodeListing& materialNodes, double time,
                                   const std::string& scenePath)
{
    return SceneError::OK;
}

int GIAlbedoMat::InnerId(uint32_t materialId) const
{
    return 0;
}

// Material Batch Implementations
template class GPUMaterialBatch<TracerBasic,
                                GIAlbedoMat,
                                GPUPrimitiveTriangle,
                                BasicSurfaceFromTri>;

template class GPUMaterialBatch<TracerBasic,
                                GIAlbedoMat,
                                GPUPrimitiveSphere,
                                BasicSurfaceFromSphr>;