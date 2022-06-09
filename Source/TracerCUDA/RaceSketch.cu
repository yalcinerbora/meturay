#include "RaceSketch.cuh"
#include "PathNode.cuh"

#include "RayLib/CoordinateConversion.h"
#include "RayLib/ColorConversion.h"
#include "CudaSystem.h"
#include "CudaSystem.hpp"

__global__ CUDA_LAUNCH_BOUNDS_1D
void KCHashRadianceToSketch(RaceSketchGPU sketchGPU,
                            // Input
                            const PathGuidingNode* gPathNodes,
                            uint32_t nodeCount,
                            uint32_t maxPathNodePerRay)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < nodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        const uint32_t nodeIndex = threadId;
        const uint32_t pathStartIndex = nodeIndex / maxPathNodePerRay * maxPathNodePerRay;

        PathGuidingNode gPathNode = gPathNodes[nodeIndex];
        // Skip if this node cannot calculate wi (last node in the path chain)
        // NEE rays or the last ray that hit light already accumulated its contibution
        // in the light hit shader
        if(!gPathNode.HasNext()) continue;
        // Don't save the incoming camera radiance
        if(nodeIndex == pathStartIndex) continue;

        // TEST
        //// Only save the incoming camera radiance
        //if(nodeIndex != pathStartIndex) continue;

        //// Only save the first bounce
        //if(nodeIndex > (pathStartIndex + 1)) continue;

        // Convert to Y color space
        float luminance = Utility::RGBToLuminance(gPathNode.totalRadiance);


        //// TEST
        //if(luminance < 0.9f) continue;

        if(luminance == 0.0f) continue;
        // TEST Firefly elimination
        if(luminance >= 30.0f) continue;

        // Generate Spherical Coordinates
        Vector3f wi = gPathNode.Wi<PathGuidingNode>(gPathNodes, pathStartIndex);
        Vector3 dirZUp = Vector3(wi[2], wi[0], wi[1]);
        Vector2f sphrCoords = Utility::CartesianToSphericalUnit(dirZUp);
        // Add to the hash table
        sketchGPU.AtomicAddData(gPathNode.worldPosition, sphrCoords, luminance);
    }
}

RaceSketchCPU::RaceSketchCPU(uint32_t numHash,
                             uint32_t numPartition,
                             float bucketWidth,
                             uint32_t seed)
    : hashesCPU(numHash, bucketWidth, seed)
    //: hashesCPU(numHash, seed)
{
    // Allocate the sketch
    float* dSketch;
    double* dTotal;
    GPUMemFuncs::AllocateMultiData(std::tie(dSketch, dTotal),
                                   mem,
                                   {numHash * numPartition, 1});


    CUDA_CHECK(cudaMemset(dTotal, 0x00, sizeof(double)));
    CUDA_CHECK(cudaMemset(dSketch, 0x00, numHash * numPartition * sizeof(float)));

    sketchGPU.gSketch = dSketch;
    sketchGPU.gTotalCount = dTotal;
    sketchGPU.hashes = hashesCPU.HashGPU();
    sketchGPU.numHash = numHash;
    sketchGPU.numPartition = numPartition;
}

void RaceSketchCPU::HashRadianceAsPhotonDensity(const PathGuidingNode* dPGNodes,
                                                uint32_t totalNodeCount,
                                                uint32_t maxPathNodePerRay,
                                                const CudaSystem& system)
{
    // Directly call the appropriate kernel
    const CudaGPU& bestGPU = system.BestGPU();
    bestGPU.GridStrideKC_X(0, (cudaStream_t)0, totalNodeCount,
                           //
                           KCHashRadianceToSketch,
                           //
                           sketchGPU,
                           dPGNodes,
                           totalNodeCount,
                           maxPathNodePerRay);
    bestGPU.WaitMainStream();
}

void RaceSketchCPU::GetSketchToCPU(std::vector<uint32_t>& sketchList,
                                   uint64_t& totalSamples) const
{
    sketchList.resize(sketchGPU.numHash * sketchGPU.numPartition);

    CUDA_CHECK(cudaMemcpy(sketchList.data(), sketchGPU.gSketch,
                          sizeof(uint32_t) * sketchList.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&totalSamples, sketchGPU.gTotalCount,
                          sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));
}