#include "GPUCameraSpherical.cuh"
#include "CudaSystem.hpp"
#include "RayLib/MemoryAlignment.h"

__global__
void KCConstructGPUCameraSpherical(GPUCameraSpherical* gCameraLocations,
                                   //
                                   const CPUCameraGroupSpherical::Data* gData,
                                   //
                                   const uint16_t* gMediumIndices,
                                   const HitKey* gWorkKeys,
                                   const TransformId* gTransformIds,
                                   //
                                   const GPUTransformI** gTransforms,
                                   uint32_t camCaount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < camCaount;
        globalId += blockDim.x * gridDim.x)
    {
        CPUCameraGroupSpherical::Data data = gData[globalId];
        new (gCameraLocations + globalId) GPUCameraSpherical(data.pixelRatio,
                                                             data.position,
                                                             data.direction,
                                                             data.up,
                                                             data.nearFar,
                                                             // Base class
                                                             gMediumIndices[globalId],
                                                             gWorkKeys[globalId],
                                                             *gTransforms[gTransformIds[globalId]]);
    }
}

SceneError CPUCameraGroupSpherical::InitializeGroup(const EndpointGroupDataList& cameraNodes,
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
        const auto direction = node.node->CommonVector3(DIR_NAME);
        const auto nearFar = node.node->CommonVector2(PLANES_NAME);
        const auto pixRatio = node.node->CommonFloat(PIX_RATIO_NAME);

        Data data = {};
        data.position = position;
        data.up = up;
        data.direction = direction;
        data.nearFar = nearFar;
        data.pixelRatio = pixRatio;

        hCameraData.push_back(data);
    }

    return SceneError::OK;
}

SceneError CPUCameraGroupSpherical::ChangeTime(const NodeListing& lightNodes, double time,
                                               const std::string& scenePath)
{
    return SceneError::CAMERA_TYPE_INTERNAL_ERROR;
}

TracerError CPUCameraGroupSpherical::ConstructEndpoints(const GPUTransformI** dGlobalTransformArray,
                                                        const CudaSystem& system)
{
    // Gen Temporary Memory
    DeviceMemory tempMemory;

    const uint16_t* dMediumIndices;
    const TransformId* dTransformIds;
    const HitKey* dWorkKeys;
    const Data* dData;
    DeviceMemory::AllocateMultiData(std::tie(dMediumIndices, dTransformIds,
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
                       KCConstructGPUCameraSpherical,
                       //
                       const_cast<GPUCameraSpherical*>(dGPUCameras),
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