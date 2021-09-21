#include "GPUCameraSpherical.cuh"
#include "CudaSystem.hpp"
#include "RayLib/MemoryAlignment.h"

__global__
void KCConstructGPUCameraSpherical(GPUCameraSpherical* gCameraLocations,
                                   //
                                   const CPUCameraGroupSpherical::Data* gData,
                                   //
                                   const TransformId* gTransformIds,
                                   const uint16_t* gMediumIndices,
                                   const HitKey* gCameraMaterialIds,
                                   //
                                   const GPUTransformI** gTransforms,
                                   uint32_t lightCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < lightCount;
        globalId += blockDim.x * gridDim.x)
    {
        CPUCameraGroupSpherical::Data data = gData[globalId];

        new (gCameraLocations + globalId) GPUCameraSpherical(data.pixelRatio,
                                                             data.position,
                                                             data.direction,
                                                             data.up,
                                                             data.nearFar,
                                                             *gTransforms[gTransformIds[globalId]],
                                                             //
                                                             gMediumIndices[globalId],
                                                             gCameraMaterialIds[globalId]);
    }
}

SceneError CPUCameraGroupSpherical::InitializeGroup(const CameraGroupDataList& cameraNodes,
                                                    const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                    const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                    uint32_t cameraMaterialBatchId,
                                                    double time,
                                                    const std::string& scenePath)
{
    cameraCount = static_cast<uint32_t>(cameraNodes.size());
    hHitKeys.reserve(cameraCount);
    hMediumIds.reserve(cameraCount);
    hTransformIds.reserve(cameraCount);
    hCameraData.reserve(cameraCount);

    uint32_t innerIndex = 0;
    for(const auto& node : cameraNodes)
    {
        // Convert Ids to inner index
        uint16_t mediumIndex = static_cast<uint16_t>(mediumIdIndexPairs.at(node.mediumId));
        uint32_t transformIndex = transformIdIndexPairs.at(node.transformId);
        HitKey materialKey = HitKey::CombinedKey(cameraMaterialBatchId,
                                                 innerIndex);

        const auto positions = node.node->AccessVector3(NAME_POSITION);
        const auto ups = node.node->AccessVector3(NAME_UP);
        const auto directions = node.node->AccessVector3(NAME_DIR);
        const auto nearFar = node.node->AccessVector2(NAME_PLANES);
        const auto pixRatios = node.node->AccessFloat(NAME_PIX_RATIO);
        assert(positions.size() == 1);
        assert(ups.size() == 1);
        assert(directions.size() == 1);
        assert(nearFar.size() == 1);
        assert(pixRatios.size() == 1);
        assert(node.node->IdCount() == 1);

        Data data = {};
        data.position = positions[0];
        data.up = ups[0];
        data.direction = directions[0];
        data.nearFar = nearFar[0];
        data.pixelRatio = pixRatios[0];

        // TODO: Fix
        visorCameraList.push_back(VisorCamera
                                  {
                                      mediumIndex,
                                      materialKey,
                                      positions[0] + directions[0],
                                      nearFar[0][0],
                                      positions[0],
                                      nearFar[0][1],
                                      ups[0],
                                      0.0f,
                                      Vector2f(0.0f, 0.0f)
                                  });

        // Load to host memory
        hHitKeys.push_back(materialKey);
        hMediumIds.push_back(mediumIndex);
        hTransformIds.push_back(transformIndex);
        hCameraData.push_back(data);
        innerIndex++;
    }

    // Allocate for GPULight classes
    size_t totalClassSize = sizeof(GPUCameraSpherical) * cameraCount;
    totalClassSize = Memory::AlignSize(totalClassSize);

    DeviceMemory::EnlargeBuffer(memory, totalClassSize);

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(memory);
    dGPUCameras = reinterpret_cast<const GPUCameraSpherical*>(dBasePtr + offset);
    offset += totalClassSize;
    assert(totalClassSize == offset);

    return SceneError::OK;
}

SceneError CPUCameraGroupSpherical::ChangeTime(const NodeListing& lightNodes, double time,
                                               const std::string& scenePath)
{
    return SceneError::CAMERA_TYPE_INTERNAL_ERROR;
}

TracerError CPUCameraGroupSpherical::ConstructCameras(const CudaSystem& system,
                                                      const GPUTransformI** dGlobalTransformArray)
{
    // Gen Temporary Memory
    DeviceMemory tempMemory;
    // Allocate for GPULight classes
    size_t matKeySize = sizeof(HitKey) * cameraCount;
    matKeySize = Memory::AlignSize(matKeySize);
    size_t mediumSize = sizeof(uint16_t) * cameraCount;
    mediumSize = Memory::AlignSize(mediumSize);
    size_t transformIdSize = sizeof(TransformId) * cameraCount;
    transformIdSize = Memory::AlignSize(transformIdSize);
    size_t dataSize = sizeof(Data) * cameraCount;
    dataSize = Memory::AlignSize(dataSize);

    size_t totalSize = (matKeySize +
                        mediumSize +
                        transformIdSize +
                        dataSize);
    DeviceMemory::EnlargeBuffer(tempMemory, totalSize);

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(tempMemory);
    const HitKey* dCameraMaterialIds = reinterpret_cast<const HitKey*>(dBasePtr + offset);
    offset += matKeySize;
    const uint16_t* dMediumIndices = reinterpret_cast<const uint16_t*>(dBasePtr + offset);
    offset += mediumSize;
    const TransformId* dTransformIds = reinterpret_cast<const TransformId*>(dBasePtr + offset);
    offset += transformIdSize;
    const Data* dDatas = reinterpret_cast<const Data*>(dBasePtr + offset);
    offset += dataSize;
    assert(totalSize == offset);

    // Set a GPU
    const CudaGPU& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    // Load Data to Temp Memory
    CUDA_CHECK(cudaMemcpy(const_cast<HitKey*>(dCameraMaterialIds),
                          hHitKeys.data(),
                          sizeof(HitKey) * cameraCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<uint16_t*>(dMediumIndices),
                          hMediumIds.data(),
                          sizeof(uint16_t) * cameraCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<TransformId*>(dTransformIds),
                          hTransformIds.data(),
                          sizeof(TransformId) * cameraCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<Data*>(dDatas),
                          hCameraData.data(),
                          sizeof(Data) * cameraCount,
                          cudaMemcpyHostToDevice));

    // Call allocation kernel
    gpu.GridStrideKC_X(0, 0,
                       CameraCount(),
                       //
                       KCConstructGPUCameraSpherical,
                       //
                       const_cast<GPUCameraSpherical*>(dGPUCameras),
                       //
                       dDatas,
                       //
                       dTransformIds,
                       dMediumIndices,
                       dCameraMaterialIds,
                       //
                       dGlobalTransformArray,
                       CameraCount());

    gpu.WaitMainStream();

    // Generate transform list
    for(uint32_t i = 0; i < CameraCount(); i++)
    {
        const auto* ptr = static_cast<const GPUCameraI*>(dGPUCameras + i);
        gpuCameraList.push_back(ptr);
    }
    return TracerError::OK;
}