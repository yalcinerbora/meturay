#include "GPUCameraPinhole.cuh"

__global__ void KCConstructPinholeCamera()
{

}

SceneError CPUCameraPinhole::InitializeGroup(const NodeListing& lightNodes,
                                             const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                             const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                             const MaterialKeyListing& allMaterialKeys,
                                             double time,
                                             const std::string& scenePath)
{
    return SceneError::CAMERA_TYPE_INTERNAL_ERROR;
}

SceneError CPUCameraPinhole::ChangeTime(const NodeListing& lightNodes, double time,
                                        const std::string& scenePath)
{
    return SceneError::CAMERA_TYPE_INTERNAL_ERROR;
}

TracerError CPUCameraPinhole::ConstructCameras(const CudaSystem& system)
{
    return TracerError::OK;
    //TracerError::;
}