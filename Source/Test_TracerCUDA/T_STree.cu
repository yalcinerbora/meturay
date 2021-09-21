#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <numeric>
#include <random>

#include "TracerCUDA/STree.cuh"
#include "TracerCUDA/STreeKC.cuh"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/DTreeKC.cuh"

#include "RayLib/CPUTimer.h"

#include "TracerCUDA/TracerDebug.h"

__global__
static void KCFindNearestDTree(uint32_t* dTreeIndices,
                               const Vector3f* gWorldPositions,
                               const STreeGPU& gSTree,
                               uint32_t totalSampleCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < totalSampleCount;
        globalId += blockDim.x * gridDim.x)
    {
        uint32_t dTreeId;
        gSTree.AcquireNearestDTree(dTreeId, gWorldPositions[globalId]);
        dTreeIndices[globalId] = dTreeId;
    }
}

TEST(PPG_STree, Empty)
{
    static constexpr Vector3f WORLD_MIN = Vector3f(10.0f, 10.0f, 10.0f);
    static constexpr Vector3f WORLD_MAX = Vector3f(-10.0f, -10.0f, -10.0f);
    static const AABB3f WorldAABB = AABB3f(WORLD_MIN, WORLD_MAX);
    static constexpr uint32_t SAMPLE_COUNT = 100;
    // RNG
    std::mt19937 rng;
    rng.seed(0);
    // Cuda System
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    // Default Constructed STree
    STree tree(WorldAABB);
    
    DeviceMemory outIndices(sizeof(uint32_t) * SAMPLE_COUNT);
    DeviceMemory inWorldPositions(sizeof(Vector3f) * SAMPLE_COUNT);

    // Do some random accesses to find a tree
    std::vector<Vector3f> worldSamples(SAMPLE_COUNT);
    for(Vector3f& worldPos : worldSamples)
    {
        worldPos = WorldAABB.Min() + WorldAABB.Span() * Vector3f(rng(), rng(), rng());           
    }
    CUDA_CHECK(cudaMemcpy(static_cast<Vector3f*>(inWorldPositions),
                          worldSamples.data(),
                          sizeof(Vector3f) * SAMPLE_COUNT,
                          cudaMemcpyHostToDevice));


    const CudaGPU& bestGPU = system.BestGPU();

    const STreeGPU* dSTreeGPU;
    const DTreeGPU** dReadDTrees;
    DTreeGPU** dWriteDTrees;
    tree.TreeGPU(dSTreeGPU, dReadDTrees, dWriteDTrees);
    bestGPU.GridStrideKC_X(0, 0, SAMPLE_COUNT,
                           //
                           KCFindNearestDTree,
                           //
                           static_cast<uint32_t*>(outIndices),
                           static_cast<Vector3f*>(inWorldPositions),
                           *dSTreeGPU,
                           SAMPLE_COUNT);

    std::vector<uint32_t> indicesCPU(SAMPLE_COUNT);
    CUDA_CHECK(cudaMemcpy(indicesCPU.data(),
                          static_cast<uint32_t*>(outIndices),
                          sizeof(uint32_t) * SAMPLE_COUNT,
                          cudaMemcpyDeviceToHost));

    for(uint32_t index : indicesCPU)
    {
        EXPECT_EQ(0, index);
    }
}

TEST(PPG_STree, Split)
{
    static constexpr Vector3f WORLD_MIN = Vector3f(10.0f, 10.0f, 10.0f);
    static constexpr Vector3f WORLD_MAX = Vector3f(-10.0f, -10.0f, -10.0f);
    static const AABB3f WorldAABB = AABB3f(WORLD_MIN, WORLD_MAX);
    // Maximum of 5 depth for each DTree
    static constexpr uint32_t D_MAX_DEPT = 50;
    // Split a DTree when it reaches more than %10 of total energy
    static constexpr float D_FLUX_SPLIT = 0.001f;
    // Split a STree leaf when it reaches 100 samples
    static constexpr uint32_t S_SPLIT = 100;
    // 
    static constexpr uint32_t ITERATION_COUNT = 15;
    static constexpr uint32_t PATH_PER_ITERATION = 100'000;
    static constexpr uint32_t RAY_COUNT = 10'000;
    static constexpr uint32_t PATH_PER_RAY = PATH_PER_ITERATION / RAY_COUNT;
    //
    static constexpr Vector3f MAX_TOTAL_RADIANCE = Vector3f(1, 1, 1);
    // Cuda System
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    // Rng
    std::mt19937 rng;
    rng.seed(0);
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

    // GPU Buffers
    DeviceMemory pathNodeMemory(PATH_PER_ITERATION * sizeof(PathGuidingNode));
    PathGuidingNode* dPathNodes = static_cast<PathGuidingNode*>(pathNodeMemory);

    STree testTree(WorldAABB);
    std::vector<PathGuidingNode> paths(PATH_PER_ITERATION);
    for(uint32_t iCount = 0; iCount < ITERATION_COUNT; iCount++)
    {
        std::uniform_int_distribution<uint32_t> treeCountDist(0, testTree.TotalTreeCount() - 1);
        // Generate Random Paths
        for(size_t i = 0; i < PATH_PER_ITERATION; i++)
        {
            uint32_t localIndex = i % PATH_PER_RAY;
            uint32_t prev = (localIndex == 0) ? PathGuidingNode::InvalidIndex : localIndex - 1;
            uint32_t next = (localIndex == (PATH_PER_RAY - 1)) ? PathGuidingNode::InvalidIndex : localIndex + 1;

            Vector3f worldUniform(uniformDist(rng), uniformDist(rng), uniformDist(rng));
            Vector3f radianceUniform(uniformDist(rng), uniformDist(rng), uniformDist(rng));

            PathGuidingNode p;
            p.worldPosition = WORLD_MIN + worldUniform * WorldAABB.Span();
            p.prevNext = Vector<2, PathGuidingNode::IndexType>(prev, next);
            p.totalRadiance = radianceUniform * MAX_TOTAL_RADIANCE;
            // Unnecessary Data for this operation
            p.nearestDTreeIndex = treeCountDist(rng);
            p.radFactor = Zero3;
            paths[i] = p;
        }

        // Add Paths to GPU
        // Copy Vertices to the GPU
        CUDA_CHECK(cudaMemcpy(dPathNodes, paths.data(),
                              PATH_PER_ITERATION * sizeof(PathGuidingNode),
                              cudaMemcpyHostToDevice));

        //// DEBUGGING
        //Utility::CPUTimer t;
        //STreeGPU treeGPU;
        //std::vector<STreeNode> nodes;
        //testTree.GetTreeToCPU(treeGPU, nodes);
        //Debug::DumpMemToFile("BS-Nodes", nodes.data(), nodes.size());
        // Accumulate Radiances
        //t.Start();
        testTree.AccumulateRaidances(dPathNodes, PATH_PER_ITERATION, PATH_PER_RAY, system);
        system.SyncAllGPUs();
        //t.Lap();
        //METU_LOG("Accum-Rad {:f}", t.Elapsed<CPUTimeSeconds>());

        // Split and Swap trees
        testTree.SplitAndSwapTrees(S_SPLIT, D_FLUX_SPLIT, D_MAX_DEPT, system);
        system.SyncAllGPUs();

        //// DEBUGGING
        //t.Lap();
        //METU_LOG("Split&Swap {:f}", t.Elapsed<CPUTimeSeconds>());
        //testTree.GetTreeToCPU(treeGPU, nodes);
        //Debug::DumpMemToFile("AS-Nodes", nodes.data(), nodes.size());
        //METU_LOG("iter {:d}", iCount);
        //METU_LOG("-----------------------------------------------------");
    }

    //// DEBUGGING
    //STreeGPU treeGPU;
    //std::vector<STreeNode> nodes;
    //testTree.GetTreeToCPU(treeGPU, nodes);
    //Debug::DumpMemToFile("Final Nodes", nodes.data(), nodes.size());
}