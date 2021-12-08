#include "SurfaceSVO.cuh"
#include "CudaSystem.hpp"

#include "RayLib/BitManipulation.h"

__global__ static
void KCMarkReduction()
{

}

SurfaceSVOCPU::SurfaceSVOCPU()
    : svoGPU{nullptr, 0u, ZeroAABB3f}
    , nodeCount(0)
    , leafCount(0)
{}

void SurfaceSVOCPU::Construct(const AcceleratorBatchMap& sceneAccelerators,
                              const AABB3f& sceneExtents,
                              float normalAngleThreshold, float areaThreshold,
                              const CudaSystem&)
{
    // Determine the actual area treshold
    // (SVO subdivides the volume by two for each child)
    Vector3f sceneSpan = sceneExtents.Span();
    float maxDim = sceneSpan[sceneSpan.Max()];

    uint32_t ratio = static_cast<uint32_t>(std::ceil(maxDim / areaThreshold));
    uint32_t depth = Utility::NextPowOfTwo(ratio);


    // Get Area, Center & Normal for each primitive
    DeviceMemory tempMemory;
    Vector3f* dNormals;
    Vector3f* dPositions;
    float* dAreas;
    // Determine total primitive count of the scene
    // TODO:
    size_t totalPrimCount = 0;

    // Allocate
    GPUMemFuncs::AllocateMultiData(std::tie(dNormals, dPositions, dAreas),
                                   tempMemory,
                                   {totalPrimCount, totalPrimCount, totalPrimCount});

    // Ask accelerators to generate data
    // TODO:


    // Subdivide the primitives (if area is too large)
    // Refine the tree
    // Merge nodes that have too small of a size
    // Factor in the normal angle treshold

    // Compact the size of the tree (due to refinement)

    // All Done!
}