#include "GPULightSkySphere.cuh"
#include "CudaSystem.hpp"

#include "RayLib/MemoryAlignment.h"
#include "RayLib/ColorConversion.h"

__global__ void KCConstructGPULightSkySphere(GPULightSkySphere* gLightLocations,
                                             //
                                             const GPUDistPiecewiseConst2D* gLuminanceDistributions,
                                             //
                                             const TextureRefI<2, Vector3f>** gRads,
                                             const uint16_t* gMediumIndices,
                                             const HitKey* gWorkKeys,
                                             const TransformId* gTransformIds,
                                             //
                                             const GPUTransformI** gTransforms,
                                             uint32_t lightCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < lightCount;
        globalId += blockDim.x * gridDim.x)
    {
        new (gLightLocations + globalId) GPULightSkySphere(gLuminanceDistributions[globalId],
                                                           //
                                                           *gRads[globalId],
                                                           gMediumIndices[globalId],
                                                           gWorkKeys[globalId],
                                                           *gTransforms[gTransformIds[globalId]]);
    }
}

__global__
void KCRGBTextureToLuminanceArray(float* gOutLuminance,
                                  const TextureRefI<2, Vector3>** gTextureRefs,
                                  uint32_t lightIndex,
                                  const Vector2ui dimension)
{
    const TextureRefI<2, Vector3>* gTextureRef = gTextureRefs[lightIndex];
    uint32_t totalWorkCount = dimension[0] * dimension[1];
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalWorkCount;
        threadId += (blockDim.x * gridDim.x))
    {
        Vector2ui id2D = Vector2ui(threadId % dimension[0],
                                   threadId / dimension[0]);
        // Convert to UV coordinates
        Vector2f invDim = Vector2f(1.0f) / Vector2f(dimension);
        Vector2f uv = Vector2f(id2D) * invDim;
        // Bypass linear interp
        uv += Vector2f(0.5f) * invDim;

        Vector3 rgb = (*gTextureRef)(uv);
        float luminance = Utility::RGBToLuminance(rgb);
        gOutLuminance[threadId] = luminance;
    }
}

SceneError CPULightGroupSkySphere::InitializeGroup(const EndpointGroupDataList& lightNodes,
                                                   const TextureNodeMap& textures,
                                                   const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                   const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                   uint32_t batchId, double time,
                                                   const std::string& scenePath)
{
    SceneError e = SceneError::OK;

    if((e = InitializeCommon(lightNodes, textures,
                             mediumIdIndexPairs,
                             transformIdIndexPairs,
                             batchId, time,
                             scenePath)) != SceneError::OK)
        return e;


    // Allocate Distribution Memory
    DeviceMemory::EnlargeBuffer(gpuDsitributionMem,
                                sizeof(GPUDistPiecewiseConst2D) * lightCount);
    dGPUDistributions = static_cast<GPUDistPiecewiseConst2D*>(gpuDsitributionMem);

    return SceneError::OK;
}

SceneError CPULightGroupSkySphere::ChangeTime(const NodeListing& lightNodes,
                                              double time,
                                              const std::string& scenePath)
{
    // TODO: Implement
    return SceneError(SceneError::LIGHT_TYPE_INTERNAL_ERRROR,
                      "Change Time function on \"CPULightGroupSkySphere\""
                      "is not yet implemented");
}

TracerError CPULightGroupSkySphere::ConstructEndpoints(const GPUTransformI** dGlobalTransformArray,
                                                       const CudaSystem& system)
{
    TracerError e = TracerError::OK;
    // Construct Texture References
    if((e = ConstructTextureReferences()) != TracerError::OK)
        return e;

    // TODO: We go to GPU -> CPU -> GPU here
    // normaly distribution data were coming from another class
    // Rewrite PWDistributions to be contructed directly from the GPUMemory as well
    std::vector<std::vector<float>> hLuminances;
    std::vector<Vector2ui> hLuminanceSizes;
    hLuminances.reserve(lightCount);
    hLuminanceSizes.reserve(lightCount);

    // Iterate over each light to generate
    DeviceMemory luminanceBuffer;
    for(uint32_t lightIndex = 0; lightIndex < lightCount; lightIndex++)
    {
        uint32_t texId = textureIdList[lightIndex];
        Vector2ui dim = (texId == std::numeric_limits<uint32_t>::max())
                            ? dim = Vector2ui(1u)
                            : dim = dTextureMemory.at(texId)->Dimensions();
        uint32_t totalCount = dim[0] * dim[1];

        DeviceMemory::EnlargeBuffer(luminanceBuffer, totalCount * sizeof(float));
        float* dLumArray = static_cast<float*>(luminanceBuffer);

        // Use your own gpu since texture resides there
        gpu.GridStrideKC_X
        (
            0, (cudaStream_t)0, totalCount,
             // Kernel
            KCRGBTextureToLuminanceArray,
            // Args
            dLumArray,
            dRadiances,
            lightIndex,
            dim
        );

        hLuminances.emplace_back(totalCount);
        CUDA_CHECK(cudaMemcpy(hLuminances.back().data(),
                              dLumArray, totalCount * sizeof(float),
                              cudaMemcpyDeviceToHost));

        hLuminanceSizes.push_back(dim);
    }

    // Construct Distribution Data
    std::vector<bool> factorInSpherical(hLuminances.size(), true);
    hLuminanceDistributions = CPUDistGroupPiecewiseConst2D(hLuminances, hLuminanceSizes,
                                                           factorInSpherical, system);

    // As a madlad directly copy the CPU residing GPU class to the GPU memory
    CUDA_CHECK(cudaMemcpy(const_cast<GPUDistPiecewiseConst2D*>(dGPUDistributions),
                          hLuminanceDistributions.DistributionGPU().data(),
                          sizeof(GPUDistPiecewiseConst2D) * lightCount,
                          cudaMemcpyHostToDevice));

    // Gen Temporary Memory
    DeviceMemory tempMemory;
    const uint16_t* dMediumIndices;
    const TransformId* dTransformIds;
    const HitKey* dWorkKeys;
    DeviceMemory::AllocateMultiData(std::tie(dMediumIndices, dTransformIds, dWorkKeys),
                                    tempMemory,
                                    {lightCount, lightCount, lightCount});

    // Set a GPU
    const CudaGPU& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    CUDA_CHECK(cudaMemcpy(const_cast<uint16_t*>(dMediumIndices),
               hMediumIds.data(),
               sizeof(uint16_t) * lightCount,
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<TransformId*>(dTransformIds),
               hTransformIds.data(),
               sizeof(TransformId) * lightCount,
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<HitKey*>(dWorkKeys),
                          hWorkKeys.data(),
                          sizeof(HitKey) * lightCount,
                          cudaMemcpyHostToDevice));

    // Call allocation kernel
    gpu.GridStrideKC_X(0, 0,
                       lightCount,
                       //
                       KCConstructGPULightSkySphere,
                       //
                       const_cast<GPULightSkySphere*>(dGPULights),
                       //
                       dGPUDistributions,
                       //
                       dRadiances,
                       dMediumIndices,
                       dWorkKeys,
                       dTransformIds,
                       //
                       dGlobalTransformArray,
                       lightCount);

    gpu.WaitMainStream();

    SetLightLists();

    return TracerError::OK;
}