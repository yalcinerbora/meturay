#include "BasicMaterials.cuh"

SceneError ConstantMat::InitializeGroup(const NodeListing& materialNodes, double time,
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

    dData = AlbedoMatData{dAlbedo};
    return SceneError::OK;
}

SceneError ConstantMat::ChangeTime(const NodeListing& materialNodes, double time,
                                   const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::NO_LOGIC_FOR_MATERIAL;
}

int ConstantMat::InnerId(uint32_t materialId) const
{
    return innerIds.at(materialId);
}

//// Material Batches
//template class GPUMaterialBatch<TracerBasic,
//                                GPUEventEstimatorEmpty,
//                                BasicMat,
//                                GPUPrimitiveEmpty,
//                                EmptySurfaceFromEmpty>;
//
//template class GPUMaterialBatch<TracerBasic,
//                                GPUEventEstimatorEmpty,
//                                BasicMat,
//                                GPUPrimitiveTriangle,
//                                EmptySurfaceFromTri>;
//
//template class GPUMaterialBatch<TracerBasic,
//                                GPUEventEstimatorEmpty,
//                                BasicMat,
//                                GPUPrimitiveSphere,
//                                EmptySurfaceFromSphr>;
//
//template class GPUMaterialBatch<TracerBasic,
//                                GPUEventEstimatorEmpty,
//                                BarycentricMat,
//                                GPUPrimitiveTriangle,
//                                BarySurfaceFromTri>;
//
//template class GPUMaterialBatch<TracerBasic,
//                                GPUEventEstimatorEmpty,
//                                SphericalMat,
//                                GPUPrimitiveSphere,
//                                SphrSurfaceFromSphr>;