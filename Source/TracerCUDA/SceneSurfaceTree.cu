#include "SceneSurfaceTree.cuh"
#include "GPUAcceleratorI.h"
#include "RNGSobol.cuh"
#include "RayLib/CPUTimer.h"

#include <cub/cub.cuh>
#include <numeric>

__global__
void KCPackSurfaces(SurfaceLeaf* gSurfaces,
                    const Vector3f* gPositions,
                    const Vector3f* gNormals,
                    uint32_t surfaceCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < surfaceCount; globalId += blockDim.x * gridDim.x)
    {
        gSurfaces[globalId] = SurfaceLeaf{gPositions[globalId],
                                          gNormals[globalId]};
    }
}

TracerError SceneSurfaceTree::Construct(const AcceleratorBatchMap& sceneAccelerators,
                                        float normalAngleThreshold,
                                        uint32_t samplePointCount,
                                        uint32_t seed,
                                        const CudaSystem& cudaSystem)
{
    Utility::CPUTimer timer;
    timer.Start();

    TracerError err = TracerError::OK;
    const CudaGPU& gpu = cudaSystem.BestGPU();
    // Generate A temp RNG for this gpu
    // TODO: implement a low-discrepancy sampler and use it here
    RNGSobolCPU rngSobol(0, cudaSystem);

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
        acceleratorAreas.push_back(acc->TotalApproximateArea(cudaSystem));
    }

    float totalArea = std::reduce(acceleratorAreas.cbegin(),
                                  acceleratorAreas.cend(),
                                  0.0f);
    std::transform(acceleratorAreas.begin(), acceleratorAreas.end(),
                   acceleratorAreas.begin(), [&](float a)
                   {
                       return a /= totalArea;
                   });
    // Divide the samples according to the area measures
    uint32_t acquiredSampleCount = 0;
    std::vector<uint32_t> samplePerAcceleratorGroup;
    samplePerAcceleratorGroup.reserve(sceneAccelerators.size());
    for(float normArea : acceleratorAreas)
    {
        assert(normArea > 0.0f);
        uint32_t i = static_cast<uint32_t>(static_cast<float>(samplePointCount) * normArea);
        samplePerAcceleratorGroup.push_back(i);
        acquiredSampleCount += i;
    }
    // Sprinkle the remaining values to the pools
    for(uint32_t i = 0; i < (samplePointCount - acquiredSampleCount); i++)
    {
        samplePerAcceleratorGroup[i % samplePerAcceleratorGroup.size()] += 1;
    }

    // Acquire samples from the accelerator
    uint32_t i = 0;
    uint32_t offset = 0;
    for(const auto& [id, acc] : sceneAccelerators)
    {
        uint32_t localSampleCount = samplePerAcceleratorGroup[i];
        // Skip if there is not samples for this accel
        if(localSampleCount == 0) continue;

        acc->SampleAreaWeightedPoints(dPositions + offset,
                                      dNormals + offset,
                                      rngSobol,
                                      localSampleCount,
                                      cudaSystem);

        offset += localSampleCount;
        i++;
    }
    assert(offset == samplePointCount);

    //
    DeviceMemory surfaceMemory;
    SurfaceLeaf* dSurfaceLeafs;
    GPUMemFuncs::AllocateMultiData(std::tie(dSurfaceLeafs),
                                   surfaceMemory, {samplePointCount});

    gpu.GridStrideKC_X(0, (cudaStream_t)0, samplePointCount,
                       //
                       KCPackSurfaces,
                       //
                       dSurfaceLeafs,
                       dPositions,
                       dNormals,
                       samplePointCount);

    // Clear temp memory (data is in surfaceMem now)
    tempMemory = DeviceMemory();

    // Compute an LBVH over the location
    float nThreshold = normalAngleThreshold * MathConstants::DegToRadCoef;
    nThreshold = std::cos(nThreshold);
    SurfaceDistanceFunctor df(nThreshold);
    if((err = lBVHSurface.Construct(dSurfaceLeafs, samplePointCount, df,
                                    cudaSystem)) != TracerError::OK)
        return err;

    // Print the generation timer
    cudaSystem.SyncAllGPUs();
    timer.Stop();
    METU_LOG("SurfaceTree generated in {} seconds", timer.Elapsed<CPUTimeSeconds>());

    // All Done!
    return TracerError::OK;
}

std::ostream& operator<<(std::ostream& s, const SurfaceLeaf& l)
{
    s << "[" << l.position << "] [" << l.normal << "]";
    return s;
}

template struct LBVHNode<SurfaceLeaf>;
template struct LinearBVHGPU<SurfaceLeaf,
                             SurfaceDistanceFunctor>;
template class LinearBVHCPU<SurfaceLeaf,
                            SurfaceDistanceFunctor,
                            GenSurfaceAABB>;