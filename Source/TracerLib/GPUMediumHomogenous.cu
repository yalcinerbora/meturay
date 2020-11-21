#include "GPUMediumHomogenous.cuh"
#include "CudaConstants.h"
#include "RayLib/MemoryAlignment.h"

__global__ void KCConstructGPUMediumHomogenous(GPUMediumHomogenous* gLocation,
                                               //
                                               const GPUMediumHomogenous::Data* gDataList,
                                               uint32_t indexOffset,
                                               uint32_t mediumCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < mediumCount;
        globalId += blockDim.x * gridDim.x)
    {
        new (gLocation) GPUMediumHomogenous(gDataList[globalId],
                                            indexOffset + globalId);
    }
}

SceneError CPUMediumHomogenous::InitializeGroup(const NodeListing& mediumNodes,
												double time,
												const std::string& scenePath)
{
    //std::vector<GPUMediumHomogenous>
    std::vector<GPUMediumHomogenous::Data> mediumData;

    for(const auto& node : mediumNodes)
    {
        std::vector<Vector3> nodeAbsList = node->AccessVector3(ABSORBTION);
        std::vector<Vector3> nodeScatList = node->AccessVector3(SCATTERING);
        std::vector<float> nodeIORList = node->AccessFloat(IOR);
        std::vector<float> nodePhaseList = node->AccessFloat(PHASE);

        assert(node->IdCount() == nodeAbsList.size());
        assert(nodeAbsList.size() == nodeScatList.size());
        assert(nodeScatList.size() == nodeIORList.size());
        assert(nodeIORList.size() == nodePhaseList.size());

        for(uint32_t i = 0; i < node->IdCount(); i++)
        {
            mediumData.push_back(GPUMediumHomogenous::Data
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
    size_t mediumCount = static_cast<uint32_t>(mediumData.size());
    size_t sizeOfData = sizeof(GPUMediumHomogenous::Data) * mediumCount;
    sizeOfData = Memory::AlignSize(sizeOfData);
    size_t sizeOfMediumClasses = sizeof(GPUMediumHomogenous) * mediumCount;
    sizeOfMediumClasses = Memory::AlignSize(sizeOfMediumClasses);

    size_t requiredSize = (sizeOfData + sizeOfMediumClasses);

    // Reallocate if memory is not enough
    DeviceMemory::EnlargeBuffer(memory, requiredSize);

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(memory);
    dMediumData = reinterpret_cast<GPUMediumHomogenous::Data*>(dBasePtr + offset);
    offset += sizeOfData;    
    dGPUMediums = reinterpret_cast<GPUMediumHomogenous*>(dBasePtr + offset);
    offset += sizeOfMediumClasses;
    assert(requiredSize == offset);

    // Copy
    CUDA_CHECK(cudaMemcpy(const_cast<GPUMediumHomogenous::Data*>(dMediumData),
                          mediumData.data(),
                          mediumCount * sizeof(GPUMediumHomogenous::Data),
                          cudaMemcpyHostToDevice));
   
	return SceneError::OK;
}

SceneError CPUMediumHomogenous::ChangeTime(const NodeListing& transformNodes, double time,
										   const std::string& scenePath)
{
    // TODO: Implement
	return SceneError::MEDIUM_TYPE_INTERNAL_ERROR;
}

TracerError CPUMediumHomogenous::ConstructMediums(const CudaSystem& system,
                                                  uint32_t indexStartOffset)
{
    // Call allocation kernel
    const CudaGPU& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    gpu.AsyncGridStrideKC_X(0, MediumCount(),
                            //
                            KCConstructGPUMediumHomogenous,
                            //
                            const_cast<GPUMediumHomogenous*>(dGPUMediums),
                            MediumCount(),
                            indexStartOffset);

    gpu.WaitAllStreams();

    // Generate transform list
    for(uint32_t i = 0; i < MediumCount(); i++)
    {
        const auto* ptr = static_cast<const GPUMediumI*>(dGPUMediums + i);
        gpuMediumList.push_back(ptr);
    }
    return TracerError::OK;
}