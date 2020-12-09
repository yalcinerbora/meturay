#include "GPUCameraPinhole.cuh"
#include "RayLib/MemoryAlignment.h"
#include "CudaConstants.hpp"

__device__
GPUCameraPinhole::GPUCameraPinhole(const Vector3& pos,
                                   const Vector3& gz,
                                   const Vector3& upp,
                                   const Vector2& nearFar,
                                   const Vector2& fov,
                                   const float& aperture,
                                   const GPUTransformI& transform,
                                   //
                                   uint16_t mediumId,
                                   HitKey materialKey)
    : GPUEndpointI(materialKey, mediumId)
    , position(transform.LocalToWorld(pos))
    , up(transform.LocalToWorld(upp))
    , planeSize(planeSize)
    , nearFar(nearFar)
{

    Vector3 gazePoint = transform.LocalToWorld(gz);
    float nearPlane = nearFar[0];
    float farPLane = nearFar[1];

    // Find world space window sizes
    float widthHalf = tanf(fov[0] * 0.5f) * nearPlane;
    float heightHalf = tanf(fov[1] * 0.5f) * nearPlane;

    // Camera Vector Correction
    Vector3 gaze = gazePoint - position;
    right = Cross(gaze, up).Normalize();
    up = Cross(right, gaze).Normalize();
    gaze = Cross(up, right).Normalize();

    // Camera parameters
    bottomLeft = (position
                  - right * widthHalf
                  - up * heightHalf
                  + gaze * nearPlane);

    planeSize = Vector2(widthHalf, heightHalf) * 2.0f;
}

__device__
void GPUCameraPinhole::Sample(// Output
                              float& distance,
                              Vector3& direction,
                              float& pdf,
                              // Input
                              const Vector3& sampleLoc,
                              // I-O
                              RandomGPU&) const
{
    // One
    direction = sampleLoc - position;
    distance = direction.Length();
    direction.NormalizeSelf();        
    pdf = 1.0f;
}

__device__
void GPUCameraPinhole::GenerateRay(// Output
                                   RayReg& ray,
                                   // Input
                                   const Vector2i& sampleId,
                                   const Vector2i& sampleMax,
                                   // I-O
                                   RandomGPU& rng) const
{
    printf("GeneratingRay...   ");

    // DX DY from stratfied sample
    Vector2 delta = Vector2(planeSize[0] / static_cast<float>(sampleMax[0]),
                            planeSize[1] / static_cast<float>(sampleMax[1]));

    // Create random location over sample rectangle
    float dX = GPUDistribution::Uniform<float>(rng);
    float dY = GPUDistribution::Uniform<float>(rng);
    Vector2 randomOffset = Vector2(dX, dY);
    //Vector2 randomOffset = Vector2(0.5f);

    Vector2 sampleDistance = Vector2(static_cast<float>(sampleId[0]),
                                     static_cast<float>(sampleId[1])) * delta;
    sampleDistance += (randomOffset * delta);
    Vector3 samplePoint = bottomLeft + (sampleDistance[0] * right)
                                     + (sampleDistance[1] * up);
    Vector3 rayDir = (samplePoint - position).Normalize();

    // Initialize Ray
    ray.ray = RayF(rayDir, position);
    ray.tMin = nearFar[0];
    ray.tMax = nearFar[1];
}

__device__ 
PrimitiveId GPUCameraPinhole::PrimitiveIndex() const
{
    return 0;
}

__global__ void KCConstructGPUCameraPinhole(GPUCameraPinhole* gCameraLocations,
                                            //
                                            const CPUCameraGroupPinhole::Data* gData,
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
        CPUCameraGroupPinhole::Data data = gData[globalId];

        new (gCameraLocations + globalId) GPUCameraPinhole(data.position,
                                                           data.gaze,
                                                           data.up,
                                                           data.nearFar,
                                                           data.fov,
                                                           data.aperture,
                                                           *gTransforms[gTransformIds[globalId]],
                                                           //
                                                           gCameraMaterialIds[globalId],
                                                           gMediumIndices[globalId]);
    }
}

SceneError CPUCameraGroupPinhole::InitializeGroup(const CameraGroupData& cameraNodes,
                                                  const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                  const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                  const MaterialKeyListing& allMaterialKeys,
                                                  double time,
                                                  const std::string& scenePath)
{
    cameraCount = static_cast<uint32_t>(cameraNodes.size());
    hHitKeys.reserve(cameraCount);
    hMediumIds.reserve(cameraCount);
    hTransformIds.reserve(cameraCount);
    hCameraData.reserve(cameraCount);

    for(const auto& node : cameraNodes)
    {
        // Convert Ids to inner index
        uint32_t mediumIndex = mediumIdIndexPairs.at(node.mediumId);
        uint32_t transformIndex = transformIdIndexPairs.at(node.transformId);
        HitKey materialKey = allMaterialKeys.at(std::make_pair(BaseConstants::EMPTY_PRIMITIVE_NAME,
                                                               node.materialId));

        const auto positions = node.node->AccessVector3(NAME_POSITION);
        const auto ups = node.node->AccessVector3(NAME_UP);
        const auto gazes = node.node->AccessVector3(NAME_POSITION);
        const auto nearFar = node.node->AccessVector2(NAME_PLANES);
        const auto fovs = node.node->AccessVector2(NAME_FOV);
        const auto apertureSize = node.node->AccessFloat(NAME_APERTURE);
        assert(positions.size() == 1);
        assert(ups.size() == 1);
        assert(gazes.size() == 1);
        assert(nearFar.size() == 1);
        assert(fovs.size() == 1);
        assert(apertureSize.size() == 1);
        assert(node.node->IdCount() == 1);

        Data data = {};
        data.position = positions[0];
        data.up = ups[0];
        data.gaze = gazes[0];
        data.nearFar = nearFar[0];        
        data.fov = fovs[0] * MathConstants::DegToRadCoef;
        data.aperture = apertureSize[0];

        // Load to host memory
        hHitKeys.push_back(materialKey);
        hMediumIds.push_back(mediumIndex);
        hTransformIds.push_back(transformIndex);
        hCameraData.push_back(data);
    }

    // Allocate for GPULight classes
    size_t totalClassSize = sizeof(GPUCameraPinhole) * cameraCount;
    totalClassSize = Memory::AlignSize(totalClassSize);

    DeviceMemory::EnlargeBuffer(memory, totalClassSize);

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(memory);
    dGPUCameras = reinterpret_cast<const GPUCameraPinhole*>(dBasePtr + offset);
    offset += totalClassSize;
    assert(totalClassSize == offset);

    return SceneError::OK;
}

SceneError CPUCameraGroupPinhole::ChangeTime(const NodeListing& lightNodes, double time,
                                             const std::string& scenePath)
{
    return SceneError::CAMERA_TYPE_INTERNAL_ERROR;
}

TracerError CPUCameraGroupPinhole::ConstructCameras(const CudaSystem& system,
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
                          sizeof(Vector3) * cameraCount,
                          cudaMemcpyHostToDevice));

    system.SyncGPUAll();

    // Call allocation kernel
    gpu.GridStrideKC_X(0, 0,
                       CameraCount(),
                       //
                       KCConstructGPUCameraPinhole,
                       //
                       const_cast<GPUCameraPinhole*>(dGPUCameras),
                       //
                       dDatas,
                       //
                       dTransformIds,
                       dMediumIndices,
                       dCameraMaterialIds,
                       //
                       dGlobalTransformArray,
                       CameraCount());

    //gpu.WaitAllStreams();
    system.SyncGPUAll();

    // Generate transform list
    for(uint32_t i = 0; i < CameraCount(); i++)
    {
        const auto* ptr = static_cast<const GPUCameraI*>(dGPUCameras + i);
        gpuCameraList.push_back(ptr);
    }
    return TracerError::OK;
}