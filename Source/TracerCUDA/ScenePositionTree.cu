#include "ScenePositionTree.cuh"
#include "GPUAcceleratorI.h"
#include "RNGMemory.h"

#include <cub/cub.cuh>

ScenePositionTree::ScenePositionTree()
{}

TracerError ScenePositionTree::Construct(const AcceleratorBatchMap& sceneAccelerators,
                                         float normalAngleThreshold,
                                         uint32_t samplePointCount,
                                         const CudaSystem& cudaSystem)
{
    TracerError err = TracerError::OK;
    const CudaGPU& gpu = cudaSystem.BestGPU();
    // Generate A temp RNG for this gpu
    // TODO: implement a low-discrepancy sampler and use it here
    RNGMemory rngMem(0, gpu);

    // Get Area, Center & Normal for each primitive
    DeviceMemory tempMemory;
    Vector3f* dPositions;
    Vector3f* dNormals;
    // Allocate
    GPUMemFuncs::AllocateMultiData(std::tie(dPositions, dNormals),
                                   tempMemory,
                                   {samplePointCount, samplePointCount});

    std::vector<float> acceleratorAreas;
    acceleratorAreas.reserve(sceneAccelerators.size());

    // Ask accelerators to generate data
    for(const auto& [id, acc] : sceneAccelerators)
    {
        acceleratorAreas.push_back(acc->TotalApproximateArea(gpu));
    }

    // Divide the samples according to the area measures
    std::vector<uint32_t> samplePerAcceleratorGroup;

    // Acquire samples from the accelerator
    uint32_t i = 0;
    uint32_t offset = 0;
    for(const auto& [id, acc] : sceneAccelerators)
    {
        uint32_t localSampleCount = samplePerAcceleratorGroup[i];
        acc->AcquireAreaWeightedSurfacePathces(dPositions + offset,
                                               dNormals,
                                               rngMem,
                                               localSampleCount,
                                               cudaSystem);

        offset += localSampleCount;
        i++;
    }
    assert(offset == samplePointCount);

    // TODO: add normal

    // Compute an LBVH over the location
    if((err = lBVHPoint.Construct(dPositions, samplePointCount,
                                  cudaSystem)) != TracerError::OK)
        return err;

    // All Done!
    return TracerError::OK;
}