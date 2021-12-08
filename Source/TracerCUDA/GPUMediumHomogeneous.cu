#include "GPUMediumHomogeneous.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"
#include "RayLib/MemoryAlignment.h"

__global__ void KCConstructGPUMediumHomogeneous(GPUMediumHomogeneous* gMediumLocations,
                                               //
                                                const GPUMediumHomogeneous::Data* gDataList,
                                                uint32_t indexOffset,
                                                uint32_t mediumCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < mediumCount;
        globalId += blockDim.x * gridDim.x)
    {
        new (gMediumLocations + globalId) GPUMediumHomogeneous(gDataList[globalId],
                                                               indexOffset + globalId);
    }
}

SceneError CPUMediumHomogeneous::InitializeGroup(const NodeListing& mediumNodes,
												 double time,
												 const std::string&)
{
    //std::vector<GPUMediumHomogenous>
    std::vector<GPUMediumHomogeneous::Data> mediumData;

    for(const auto& node : mediumNodes)
    {
        std::vector<Vector3> nodeAbsList = node->AccessVector3(ABSORBTION, time);
        std::vector<Vector3> nodeScatList = node->AccessVector3(SCATTERING, time);
        std::vector<float> nodeIORList = node->AccessFloat(IOR, time);
        std::vector<float> nodePhaseList = node->AccessFloat(PHASE, time);

        assert(node->IdCount() == nodeAbsList.size());
        assert(nodeAbsList.size() == nodeScatList.size());
        assert(nodeScatList.size() == nodeIORList.size());
        assert(nodeIORList.size() == nodePhaseList.size());

        for(uint32_t i = 0; i < node->IdCount(); i++)
        {
            mediumData.push_back(GPUMediumHomogeneous::Data
            {
                nodeAbsList[i],
                nodeScatList[i],
                nodeAbsList[i] + nodeScatList[i],
                nodeIORList[i],
                nodePhaseList[i]
            });
        }
    }
    // Finally Allocate and load to GPU memory
    mediumCount = static_cast<uint32_t>(mediumData.size());
    GPUMemFuncs::AllocateMultiData(std::tie(dMediumData, dGPUMediums), memory,
                                   {mediumCount, mediumCount});
    // Copy
    CUDA_CHECK(cudaMemcpy(const_cast<GPUMediumHomogeneous::Data*>(dMediumData),
                          mediumData.data(),
                          mediumCount * sizeof(GPUMediumHomogeneous::Data),
                          cudaMemcpyHostToDevice));

	return SceneError::OK;
}

SceneError CPUMediumHomogeneous::ChangeTime(const NodeListing&, double,
										    const std::string&)
{
    // TODO: Implement
	return SceneError::MEDIUM_TYPE_INTERNAL_ERROR;
}

TracerError CPUMediumHomogeneous::ConstructMediums(const CudaSystem& system,
                                                   uint32_t indexStartOffset)
{
    // Call allocation kernel
    const CudaGPU& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    gpu.GridStrideKC_X(0, 0,
                       MediumCount(),
                       //
                       KCConstructGPUMediumHomogeneous,
                       //
                       const_cast<GPUMediumHomogeneous*>(dGPUMediums),
                       dMediumData,
                       indexStartOffset,
                       MediumCount());

    gpu.WaitMainStream();

    // Generate transform list
    for(uint32_t i = 0; i < MediumCount(); i++)
    {
        const auto* ptr = static_cast<const GPUMediumI*>(dGPUMediums + i);
        gpuMediumList.push_back(ptr);
    }
    return TracerError::OK;
}