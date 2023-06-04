#include "GPUCameraPinhole.cuh"
#include "CudaSystem.hpp"
#include "RayLib/MemoryAlignment.h"

__global__ void KCConstructGPUCameraPinhole(GPUCameraPinhole* gCameraLocations,
                                            //
                                            const CPUCameraGroupPinhole::Data* gData,
                                            //
                                            const uint16_t* gMediumIndices,
                                            const HitKey* gWorkKeys,
                                            const TransformId* gTransformIds,
                                            //
                                            const GPUTransformI** gTransforms,
                                            uint32_t camCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < camCount;
        globalId += blockDim.x * gridDim.x)
    {
        CPUCameraGroupPinhole::Data data = gData[globalId];

        new (gCameraLocations + globalId) GPUCameraPinhole(data.position,
                                                           data.gaze,
                                                           data.up,
                                                           data.nearFar,
                                                           data.fov,
                                                           // Base class
                                                           gMediumIndices[globalId],
                                                           gWorkKeys[globalId],
                                                           *gTransforms[gTransformIds[globalId]]);
    }
}

SceneError CPUCameraGroupPinhole::InitializeGroup(const EndpointGroupDataList& cameraNodes,
                                                  const TextureNodeMap& textures,
                                                  const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                  const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                  uint32_t batchId, double time,
                                                  const std::string& scenePath)
{
    SceneError e = SceneError::OK;

    if((e = InitializeCommon(cameraNodes, textures,
                             mediumIdIndexPairs,
                             transformIdIndexPairs,
                             batchId, time,
                             scenePath)) != SceneError::OK)
        return e;


    hCameraData.reserve(cameraCount);
    for(const auto& node : cameraNodes)
    {
        const auto position = node.node->CommonVector3(POSITION_NAME);
        const auto up = node.node->CommonVector3(UP_NAME);
        const auto gaze = node.node->CommonVector3(GAZE_NAME);
        const auto nearFar = node.node->CommonVector2(PLANES_NAME);
        const auto fov = node.node->CommonVector2(FOV_NAME);

        Data data = {};
        data.position = position;
        data.up = up;
        data.gaze = gaze;
        data.nearFar = nearFar;

        // Correct FoV using FovY
        data.fov = fov * MathConstants::DegToRadCoef;
        float aspect = fov[0] / fov[1];
        data.fov[0] = 2.0f * std::atan(std::tan(data.fov[1] * 0.5f) * aspect);

        hCameraData.push_back(data);
    }

    return SceneError::OK;
}

SceneError CPUCameraGroupPinhole::ChangeTime(const NodeListing&, double,
                                             const std::string&)
{
    return SceneError::CAMERA_TYPE_INTERNAL_ERROR;
}

TracerError CPUCameraGroupPinhole::ConstructEndpoints(const GPUTransformI** dGlobalTransformArray,
                                                      const AABB3f&,
                                                      const CudaSystem& system)
{
    // Gen Temporary Memory
    DeviceMemory tempMemory;

    const uint16_t* dMediumIndices;
    const TransformId* dTransformIds;
    const HitKey* dWorkKeys;
    const Data* dData;
    GPUMemFuncs::AllocateMultiData(std::tie(dMediumIndices, dTransformIds,
                                            dWorkKeys, dData),
                                   tempMemory,
                                   {cameraCount, cameraCount,
                                   cameraCount, cameraCount});

    // Set a GPU
    const CudaGPU& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    // Load Data to Temp Memory
    CUDA_CHECK(cudaMemcpy(const_cast<uint16_t*>(dMediumIndices),
                          hMediumIds.data(),
                          sizeof(uint16_t) * cameraCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<TransformId*>(dTransformIds),
                          hTransformIds.data(),
                          sizeof(TransformId) * cameraCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<HitKey*>(dWorkKeys),
                          hWorkKeys.data(),
                          sizeof(HitKey) * cameraCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<Data*>(dData),
                          hCameraData.data(),
                          sizeof(Data) * cameraCount,
                          cudaMemcpyHostToDevice));

    // Call allocation kernel
    gpu.GridStrideKC_X(0, 0,
                       cameraCount,
                       //
                       KCConstructGPUCameraPinhole,
                       //
                       const_cast<GPUCameraPinhole*>(dGPUCameras),
                       //
                       dData,
                       //
                       dMediumIndices,
                       dWorkKeys,
                       dTransformIds,
                       //
                       dGlobalTransformArray,
                       cameraCount);

    gpu.WaitMainStream();

    SetCameraLists();

    return TracerError::OK;
}