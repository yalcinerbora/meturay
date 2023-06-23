#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <numeric>
#include <random>

#include "TracerCUDA/STree.cuh"
#include "TracerCUDA/STreeKC.cuh"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/DTreeKC.cuh"

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
    STree tree(WorldAABB, system);

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
    const DTreeGPU* dReadDTrees;
    DTreeGPU* dWriteDTrees;
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
        EXPECT_EQ(0u, index);
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
    static constexpr float D_FLUX_SPLIT = 0.1f;
    // Split a STree leaf when it reaches 100 samples
    static constexpr uint32_t S_SPLIT = 100;
    //
    static constexpr uint32_t ITERATION_COUNT = 150;
    static constexpr uint32_t PATH_PER_ITERATION = 200'000;
    static constexpr uint32_t RAY_COUNT = 15'000;
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
    DeviceMemory pathNodeMemory(PATH_PER_ITERATION * sizeof(PPGPathNode));
    PPGPathNode* dPathNodes = static_cast<PPGPathNode*>(pathNodeMemory);

    STree testTree(WorldAABB, system);
    std::vector<PPGPathNode> paths(PATH_PER_ITERATION);
    for(uint32_t iCount = 0; iCount < ITERATION_COUNT; iCount++)
    {
        std::uniform_int_distribution<uint32_t> treeCountDist(0, testTree.TotalTreeCount() - 1);
        // Generate Random Paths
        for(size_t i = 0; i < PATH_PER_ITERATION; i++)
        {
            uint32_t localIndex = i % PATH_PER_RAY;
            uint32_t prev = (localIndex == 0) ? PPGPathNode::InvalidIndex : localIndex - 1;
            uint32_t next = (localIndex == (PATH_PER_RAY - 1)) ? PPGPathNode::InvalidIndex : localIndex + 1;

            Vector3f worldUniform(uniformDist(rng), uniformDist(rng), uniformDist(rng));
            Vector3f radianceUniform(uniformDist(rng), uniformDist(rng), uniformDist(rng));

            PPGPathNode p;
            p.worldPosition = WORLD_MIN + worldUniform * WorldAABB.Span();
            p.prevNext = Vector<2, PPGPathNode::IndexType>(prev, next);
            p.totalRadiance = radianceUniform * MAX_TOTAL_RADIANCE;
            // Unnecessary Data for this operation
            p.dataStructIndex = treeCountDist(rng);
            p.radFactor = Zero3;
            paths[i] = p;
        }

        // Add Paths to GPU
        // Copy Vertices to the GPU
        CUDA_CHECK(cudaMemcpy(dPathNodes, paths.data(),
                              PATH_PER_ITERATION * sizeof(PPGPathNode),
                              cudaMemcpyHostToDevice));

        //// DEBUGGING
        //for(uint32_t i = 0; i < testTree.DTrees().TreeCount(); i++)
        //{
        //    using namespace std::string_literals;

        //    DTreeGPU tree;
        //    std::vector<DTreeNode> nodes;

        //    testTree.DTrees().GetReadTreeToCPU(tree, nodes, i);
        //    Debug::DumpMemToFile("BeforeSplit-readNodes"s,
        //                         nodes.data(), nodes.size(),
        //                         (i != 0));

        //    testTree.DTrees().GetWriteTreeToCPU(tree, nodes, i);
        //    Debug::DumpMemToFile("BeforeSplit-writeNodes"s,
        //                         nodes.data(), nodes.size(),
        //                         (i != 0));
        //}

        // Accumulate Radiances
        testTree.AccumulateRaidances(dPathNodes, PATH_PER_ITERATION, PATH_PER_RAY, system);
        // Split and Swap trees
        testTree.SplitAndSwapTrees(S_SPLIT, D_FLUX_SPLIT, D_MAX_DEPT, system);

        //// DEBUGGING
        //METU_LOG("After Swap TreeCount:{}, MemoryGPU:{} MiB",
        //         testTree.TotalTreeCount(),
        //         testTree.UsedGPUMemory() / 1024 / 1024);
        //for(uint32_t i = 0; i < testTree.DTrees().TreeCount(); i++)
        //{
        //    using namespace std::string_literals;

        //    DTreeGPU tree;
        //    std::vector<DTreeNode> nodes;

        //    testTree.DTrees().GetReadTreeToCPU(tree, nodes, i);
        //    Debug::DumpMemToFile("AfterSwap-readNodes"s,
        //                         nodes.data(), nodes.size(),
        //                         (i != 0));

        //    testTree.DTrees().GetWriteTreeToCPU(tree, nodes, i);
        //    Debug::DumpMemToFile("AfterSwap-writeNodes"s,
        //                         nodes.data(), nodes.size(),
        //                         (i != 0));
        //}
        //METU_LOG("-----------------------------------------------------");
    }
}