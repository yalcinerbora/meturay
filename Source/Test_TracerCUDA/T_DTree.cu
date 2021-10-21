#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <numeric>
#include <random>

#include "RayLib/ColorConversion.h"

#include "TracerCUDA/DTree.cuh"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/DTreeKC.cuh"
#include "TracerCUDA/RNGMemory.h"
#include "TracerCUDA/Random.cuh"
#include "TracerCUDA/ParallelReduction.cuh"

#include "TracerCUDA/TracerDebug.h"

using ::testing::FloatEq;

__global__
static void KCDirToCoord(Vector3f* dirs,
                         const Vector2f* coords,
                         size_t coordCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < coordCount; globalId += blockDim.x * gridDim.x)
    {
        float pdf;
        dirs[globalId] = DTreeGPU::TreeCoordsToWorldDir(pdf, coords[globalId]);
    }
}

__global__
static void KCCoordToDir(Vector2f* coords,
                         const Vector3f* dirs,
                         size_t coordCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < coordCount; globalId += blockDim.x * gridDim.x)
    {
        coords[globalId] = DTreeGPU::WorldDirToTreeCoords(dirs[globalId]);
    }
}

__global__
static void KCSampleTree(Vector3f* gDirections,
                         float* gPdfs,
                         //
                         const DTreeGPU* gDTree,
                         RNGGMem gRNGStates,

                         uint32_t sampleCount)
{
    RandomGPU rng(gRNGStates, LINEAR_GLOBAL_ID);

    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < sampleCount; globalId += blockDim.x * gridDim.x)
    {
        float pdf;
        Vector3f direction = gDTree->Sample(pdf, rng);

        gPdfs[globalId] = pdf;
        gDirections[globalId] = direction;
    }
}

__global__
static void KCPdfDivide(Vector3f* gDirections,
                        const float* gPdfs,
                        //
                        uint32_t sampleCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < sampleCount; globalId += blockDim.x * gridDim.x)
    {
        Vector3f direction = gDirections[globalId];
        float pdf = gPdfs[globalId];
        direction = (pdf == 0.0f) ? Zero3 : direction / pdf;
        gDirections[globalId] = direction;
    }
}


TEST(PPG_DTree, Empty)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    // Constants
    // If a node has %110 or more total energy, split
    // Impossible case tree should not be split
    static constexpr float FLUX_RATIO = 1.1f;
    // Maximum allowed depth of the tree
    static constexpr uint32_t DEPTH_LIMIT = 10;

    std::vector<DTreeNode> nodes;
    DTreeGPU tree;

    // Initialize Check
    DTreeGroup testTree;
    testTree.AllocateDefaultTrees(1, system);
    testTree.GetReadTreeToCPU(tree, nodes, 0);
    EXPECT_EQ(DTreeGroup::MinIrradiance * 4, tree.irradiance);
    EXPECT_EQ(0, tree.totalSamples);
    EXPECT_EQ(1, tree.nodeCount);
    EXPECT_EQ(1, nodes.size());
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[0]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[1]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[2]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[3]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().parentIndex);
    EXPECT_EQ(DTreeGroup::MinIrradiance, nodes.front().irradianceEstimates[0]);
    EXPECT_EQ(DTreeGroup::MinIrradiance, nodes.front().irradianceEstimates[1]);
    EXPECT_EQ(DTreeGroup::MinIrradiance, nodes.front().irradianceEstimates[2]);
    EXPECT_EQ(DTreeGroup::MinIrradiance, nodes.front().irradianceEstimates[3]);
    testTree.GetWriteTreeToCPU(tree, nodes, 0);
    EXPECT_EQ(0.0f, tree.irradiance);
    EXPECT_EQ(0, tree.totalSamples);
    EXPECT_EQ(1, tree.nodeCount);
    EXPECT_EQ(1, nodes.size());
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[0]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[1]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[2]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[3]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().parentIndex);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[0]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[1]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[2]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[3]);

    // After Swap Check
    testTree.SwapTrees(FLUX_RATIO, DEPTH_LIMIT, system);
    system.SyncAllGPUs();
    testTree.GetReadTreeToCPU(tree, nodes, 0);
    EXPECT_EQ(DTreeGroup::MinIrradiance * 4, tree.irradiance);
    EXPECT_EQ(0, tree.totalSamples);
    EXPECT_EQ(1, tree.nodeCount);
    EXPECT_EQ(1, nodes.size());
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[0]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[1]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[2]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[3]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().parentIndex);
    EXPECT_EQ(DTreeGroup::MinIrradiance, nodes.front().irradianceEstimates[0]);
    EXPECT_EQ(DTreeGroup::MinIrradiance, nodes.front().irradianceEstimates[1]);
    EXPECT_EQ(DTreeGroup::MinIrradiance, nodes.front().irradianceEstimates[2]);
    EXPECT_EQ(DTreeGroup::MinIrradiance, nodes.front().irradianceEstimates[3]);
    testTree.GetWriteTreeToCPU(tree, nodes, 0);
    EXPECT_EQ(0.0f, tree.irradiance);
    EXPECT_EQ(0, tree.totalSamples);
    EXPECT_EQ(1, tree.nodeCount);
    EXPECT_EQ(1, nodes.size());
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[0]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[1]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[2]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[3]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().parentIndex);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[0]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[1]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[2]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[3]);
}

TEST(PPG_DTree, AddThenSwap)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    // Constants
    // If a node has %10 or more total energy, split
    static constexpr float FLUX_RATIO = 0.001f;
    // Maximum allowed depth of the tree
    static constexpr uint32_t DEPTH_LIMIT = 10;
    static constexpr Vector3f RADIANCE = Vector3f{10.0f, 10.0f, 10.0f};

    // Check buffers
    std::vector<DTreeNode> nodes;
    DTreeGPU treeGPU;

    PathGuidingNode camNode;
    camNode.worldPosition = Vector3f{0.0f, 0.0f, 0.0f};
    camNode.prevNext = Vector<2, PathNode::IndexType>(PathNode::InvalidIndex, 1);
    camNode.radFactor = Zero3;
    camNode.nearestDTreeIndex = 0;
    camNode.totalRadiance = RADIANCE;
    PathGuidingNode midNode0;
    midNode0.worldPosition = Vector3f{10.0f, 10.0f, 0.0f};
    midNode0.prevNext = Vector<2, PathNode::IndexType>(0, 2);
    midNode0.radFactor = Zero3;
    midNode0.nearestDTreeIndex = 0;
    midNode0.totalRadiance = RADIANCE;
    PathGuidingNode midNode1;
    midNode1.worldPosition = Vector3f{0.0f, 0.0f, 0.0f};
    midNode1.prevNext = Vector<2, PathNode::IndexType>(1, 3);
    midNode1.radFactor = Zero3;
    midNode1.nearestDTreeIndex = 0;
    midNode1.totalRadiance = RADIANCE;
    PathGuidingNode midNode2;
    midNode2.worldPosition = Vector3f{-10.0f, 10.0f, 0.0f};
    midNode2.prevNext = Vector<2, PathNode::IndexType>(2, 4);
    midNode2.radFactor = Zero3;
    midNode2.nearestDTreeIndex = 0;
    midNode2.totalRadiance = RADIANCE;
    PathGuidingNode endNode;
    endNode.worldPosition = Vector3f{0.0f, 0.0f, 0.0f};
    endNode.prevNext = Vector<2, PathNode::IndexType>(3, PathNode::InvalidIndex);
    endNode.radFactor = Zero3;
    endNode.nearestDTreeIndex = 0;
    endNode.totalRadiance = RADIANCE;

    std::vector<PathGuidingNode> pathNodes =
    {
        camNode,
        midNode0,
        midNode1,
        midNode2,
        endNode
    };

    std::vector<Vector3f> directions;
    for(const PathGuidingNode& p : pathNodes)
    {
        if(p.prevNext[1] != PathNode::InvalidIndex)
        {
            directions.push_back(pathNodes[p.prevNext[1]].worldPosition - p.worldPosition);
            directions.back().NormalizeSelf();
        }
    }

    // Create Tree
    DTreeGroup testTree;
    testTree.AllocateDefaultTrees(1, system);
    // Copy Vertices to the GPU
    DeviceMemory pathNodeMemory(pathNodes.size() * sizeof(PathGuidingNode));
    PathGuidingNode* dPathNodes = static_cast<PathGuidingNode*>(pathNodeMemory);
    CUDA_CHECK(cudaMemcpy(dPathNodes, pathNodes.data(),
                          pathNodes.size() * sizeof(PathGuidingNode),
                          cudaMemcpyHostToDevice));

    // Push these values to the Tree
    const uint32_t PathNodePerRay = static_cast<uint32_t>(pathNodes.size());
    testTree.AddRadiancesFromPaths(dPathNodes,
                                   static_cast<uint32_t>(pathNodes.size()),
                                   PathNodePerRay,
                                   system);
    system.SyncAllGPUs();

    // Check Tree
    testTree.GetWriteTreeToCPU(treeGPU, nodes, 0);
    EXPECT_EQ(nodes.size(), 1);
    for(size_t i = 0; i < nodes.size(); i++)
    {
        const DTreeNode& node = nodes[i];
        if(node.parentIndex == std::numeric_limits<uint32_t>::max())
        {
            // This is root
            // Root should be the very first element
            EXPECT_EQ(0, i);
            EXPECT_EQ(treeGPU.totalSamples, pathNodes.size());
        }

        // There should be only root note and each of its values
        // should be RGB = Y of (10,10,10)
        EXPECT_TRUE(node.childIndices[0] == std::numeric_limits<uint32_t>::max());
        EXPECT_NEAR(Utility::RGBToLuminance(RADIANCE), node.irradianceEstimates[0],
                    MathConstants::VeryLargeEpsilon);
        EXPECT_TRUE(node.childIndices[1] == std::numeric_limits<uint32_t>::max());
        EXPECT_NEAR(Utility::RGBToLuminance(RADIANCE), node.irradianceEstimates[1],
                    MathConstants::VeryLargeEpsilon);
        EXPECT_TRUE(node.childIndices[2] == std::numeric_limits<uint32_t>::max());
        EXPECT_NEAR(Utility::RGBToLuminance(RADIANCE), node.irradianceEstimates[2],
                    MathConstants::VeryLargeEpsilon);
        EXPECT_TRUE(node.childIndices[3] == std::numeric_limits<uint32_t>::max());
        EXPECT_NEAR(Utility::RGBToLuminance(RADIANCE), node.irradianceEstimates[3],
                    MathConstants::VeryLargeEpsilon);
    }

    // Do the swap
    testTree.SwapTrees(FLUX_RATIO, DEPTH_LIMIT, system);
    system.SyncAllGPUs();

    // Check again
    testTree.GetReadTreeToCPU(treeGPU, nodes, 0);
    for(size_t i = 0; i < nodes.size(); i++)
    {
        const DTreeNode& node = nodes[i];
        float total = node.irradianceEstimates.Sum();
        if(node.parentIndex == std::numeric_limits<uint32_t>::max())
        {
            // This is root
            // Root should be the very first element
            EXPECT_EQ(0, i);
            EXPECT_EQ(treeGPU.totalSamples, pathNodes.size());
            EXPECT_FLOAT_EQ(treeGPU.irradiance, total);
            EXPECT_NEAR(Utility::RGBToLuminance(RADIANCE) * 4, total,
                        MathConstants::VeryLargeEpsilon);
            continue;
        }

        const DTreeNode& parent = nodes[node.parentIndex];
        uint32_t childId = UINT32_MAX;
        childId = (parent.childIndices[0] == i) ? 0 : childId;
        childId = (parent.childIndices[1] == i) ? 1 : childId;
        childId = (parent.childIndices[2] == i) ? 2 : childId;
        childId = (parent.childIndices[3] == i) ? 3 : childId;
        EXPECT_FLOAT_EQ(total, parent.irradianceEstimates[childId]);
    }
    testTree.GetWriteTreeToCPU(treeGPU, nodes, 0);
    for(size_t i = 0; i < nodes.size(); i++)
    {
        const DTreeNode& node = nodes[i];
        float total = node.irradianceEstimates.Sum();
        if(node.parentIndex == std::numeric_limits<uint32_t>::max())
        {
            // This is root
            // Root should be the very first element
            EXPECT_EQ(0, i);
            EXPECT_EQ(treeGPU.totalSamples, 0);
            EXPECT_FLOAT_EQ(treeGPU.irradiance, 0);
            EXPECT_EQ(treeGPU.nodeCount, 5);
        }
        EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[0]);
        EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[1]);
        EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[2]);
        EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[3]);
    }
}

TEST(PPG_DTree, SwapStress)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    constexpr int ITERATION_COUNT = 500;
    constexpr int PATH_PER_ITERATION = 5000;
    constexpr int RAY_COUNT = 500;
    constexpr int PATH_PER_RAY = PATH_PER_ITERATION / RAY_COUNT;

    constexpr int DTREE_ID = 0;
    constexpr Vector3f MAX_TOTAL_RADIANCE = Vector3f(1, 1, 1);
    constexpr Vector3f MIN_WORLD_BOUND = Vector3f(-10, -10, -10);
    constexpr Vector3f MAX_WORLD_BOUND = Vector3f(10, 10, 10);
    // Change depth on each iteration
    // just sto stress
    constexpr uint32_t DEPTH_MIN = 0;
    constexpr uint32_t DEPTH_MAX = 16;
    // Also change the flux
    constexpr float FLUX_MIN = 0.001f;
    constexpr float FLUX_MAX = 0.1f;

    const Vector3f worldBound = MAX_WORLD_BOUND - MIN_WORLD_BOUND;
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

    std::mt19937 rng;
    rng.seed(0);

    // GPU Buffers
    DeviceMemory pathNodeMemory(PATH_PER_ITERATION * sizeof(PathGuidingNode));
    PathGuidingNode* dPathNodes = static_cast<PathGuidingNode*>(pathNodeMemory);

    // Check buffer
    DTreeGPU treeGPU;
    std::vector<DTreeNode> nodes;

    // Stress the Tree by randomly adding data multiple times
    DTreeGroup testTree;
    testTree.AllocateDefaultTrees(1, system);
    std::vector<PathGuidingNode> paths(PATH_PER_ITERATION);
    for(int iCount = 0; iCount < ITERATION_COUNT; iCount++)
    {
        // Constants for this itertion
        // If a node has %X or more total energy, split
        const float fluxRatio = FLUX_MIN + uniformDist(rng) * (FLUX_MAX - FLUX_MIN);
        // Maximum allowed depth of the tree
        uint32_t depthLimit = DEPTH_MIN + static_cast<uint32_t>(uniformDist(rng) * (DEPTH_MAX - DEPTH_MIN));

        //METU_LOG("Depth {:d}, Flux {:f}", depthLimit, fluxRatio);

        for(size_t i = 0; i < PATH_PER_ITERATION; i++)
        {
            uint32_t localIndex = i % PATH_PER_RAY;
            uint32_t prev = (localIndex == 0) ? PathGuidingNode::InvalidIndex : localIndex - 1;
            uint32_t next = (localIndex == (PATH_PER_RAY - 1)) ? PathGuidingNode::InvalidIndex : localIndex + 1;

            Vector3f worldUniform(uniformDist(rng), uniformDist(rng), uniformDist(rng));
            Vector3f radianceUniform(uniformDist(rng), uniformDist(rng), uniformDist(rng));

            PathGuidingNode p;
            p.worldPosition = MIN_WORLD_BOUND + worldUniform * worldBound;
            p.prevNext = Vector<2, PathGuidingNode::IndexType>(prev, next);
            p.totalRadiance = radianceUniform * MAX_TOTAL_RADIANCE;
            // Unnecessary Data for this operation
            p.nearestDTreeIndex = DTREE_ID;
            p.radFactor = Zero3;
            paths[i] = p;
        }


        testTree.GetWriteTreeToCPU(treeGPU, nodes, 0);
        Debug::DumpMemToFile("nodes", nodes.data(), nodes.size());

        // Copy Vertices to the GPU
        CUDA_CHECK(cudaMemcpy(dPathNodes, paths.data(),
                              PATH_PER_ITERATION * sizeof(PathGuidingNode),
                              cudaMemcpyHostToDevice));
        // Do add radiance kernel
        testTree.AddRadiancesFromPaths(dPathNodes,
                                       PATH_PER_ITERATION,
                                       PATH_PER_RAY,
                                       system);
        system.SyncAllGPUs();

        // Check if radiance is properly added
        testTree.GetWriteTreeToCPU(treeGPU, nodes, 0);
        for(size_t i = 0; i < nodes.size(); i++)
        {
            const DTreeNode& node = nodes[i];
            if(node.parentIndex == std::numeric_limits<uint32_t>::max())
            {
                // This is root
                // Root should be the very first element
                EXPECT_EQ(0, i);
                EXPECT_EQ(treeGPU.totalSamples, PATH_PER_ITERATION);
                continue;
            }

            // Only leafs should have value
            if(node.childIndices[0] != std::numeric_limits<uint32_t>::max())
                EXPECT_EQ(0.0f, node.irradianceEstimates[0]);
            if(node.childIndices[1] != std::numeric_limits<uint32_t>::max())
                EXPECT_EQ(0.0f, node.irradianceEstimates[1]);
            if(node.childIndices[2] != std::numeric_limits<uint32_t>::max())
                EXPECT_EQ(0.0f, node.irradianceEstimates[2]);
            if(node.childIndices[3] != std::numeric_limits<uint32_t>::max())
                EXPECT_EQ(0.0f, node.irradianceEstimates[3]);
        }

        testTree.SwapTrees(fluxRatio, depthLimit, system);
        system.SyncAllGPUs();

        // Check integrity of the new write tree
        testTree.GetWriteTreeToCPU(treeGPU, nodes, 0);
        for(size_t i = 0; i < nodes.size(); i++)
        {
            const DTreeNode& node = nodes[i];
            EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[0]);
            EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[1]);
            EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[2]);
            EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[3]);

            if(node.parentIndex == std::numeric_limits<uint32_t>::max())
            {
                // This is root
                // Root should be the very first element
                EXPECT_EQ(0, i);
                EXPECT_EQ(treeGPU.totalSamples, 0);
                EXPECT_EQ(0.0f, treeGPU.irradiance);
                continue;
            }

            // Try to go to the parent
            const DTreeNode* n = &node;
            while(n->parentIndex != std::numeric_limits<uint32_t>::max())
            {
                n = &nodes[n->parentIndex];
            }
            // After back propogation
            // check if we actually reached to the parent
            ptrdiff_t index = n - nodes.data();
            EXPECT_EQ(0, index);
        }
        //// DEBUG
        //Debug::DumpMemToFile("WT", &treeGPU, 1);
        //Debug::DumpMemToFile("WTN", nodes.data(), nodes.size());

        // Check integrity of the new read tree
        testTree.GetReadTreeToCPU(treeGPU, nodes, 0);
        for(size_t i = 0; i < nodes.size(); i++)
        {
            const DTreeNode& node = nodes[i];
            float total = node.irradianceEstimates.Sum();
            if(node.parentIndex == std::numeric_limits<uint32_t>::max())
            {
                // This is root
                // Root should be the very first element
                EXPECT_EQ(0, i);
                EXPECT_EQ(treeGPU.totalSamples, PATH_PER_ITERATION);
                EXPECT_NEAR(treeGPU.irradiance, total,
                            MathConstants::VeryLargeEpsilon);
                continue;
            }

            const DTreeNode& parent = nodes[node.parentIndex];
            uint32_t childId = UINT32_MAX;
            childId = (parent.childIndices[0] == i) ? 0 : childId;
            childId = (parent.childIndices[1] == i) ? 1 : childId;
            childId = (parent.childIndices[2] == i) ? 2 : childId;
            childId = (parent.childIndices[3] == i) ? 3 : childId;
            EXPECT_FLOAT_EQ(total, parent.irradianceEstimates[childId]);
        }
        //// DEBUG
        //Debug::DumpMemToFile("RT", &treeGPU, 1);
        //Debug::DumpMemToFile("RTN", nodes.data(), nodes.size());
    }
 }

TEST(PPG_DTree, LargeToSmall)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    constexpr int ADD_ITERATION_COUNT = 50;
    constexpr int PATH_PER_ITERATION = 50;
    constexpr int RAY_COUNT = 5;
    constexpr int PATH_PER_RAY = PATH_PER_ITERATION / RAY_COUNT;

    constexpr int DTREE_ID = 0;
    constexpr Vector3f MIN_WORLD_BOUND = Vector3f(-10, -10, -10);
    constexpr Vector3f MAX_WORLD_BOUND = Vector3f(10, 10, 10);
    // DTree parameters
    constexpr uint32_t DEPTH_LIMIT = 10;
    constexpr float FLUX_RATIO = 0.005f;

    const Vector3f direction = Vector3f(1.0f).Normalize();
    const Vector3f worldBound = MAX_WORLD_BOUND - MIN_WORLD_BOUND;
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

    std::mt19937 rng;
    rng.seed(0);
    // Check buffer
    DTreeGPU treeGPU;
    std::vector<DTreeNode> nodes;
    // GPU Buffers
    DeviceMemory pathNodeMemory(PATH_PER_ITERATION * sizeof(PathGuidingNode));
    PathGuidingNode* dPathNodes = static_cast<PathGuidingNode*>(pathNodeMemory);

    // First create a deep tree
    DTreeGroup testTree;
    testTree.AllocateDefaultTrees(1, system);
    std::vector<PathGuidingNode> paths(PATH_PER_ITERATION);
    for(int iCount = 0; iCount < ADD_ITERATION_COUNT; iCount++)
    {
        for(size_t i = 0; i < PATH_PER_ITERATION; i++)
        {
            uint32_t localIndex = i % PATH_PER_RAY;
            uint32_t prev = (localIndex == 0) ? PathGuidingNode::InvalidIndex : localIndex - 1;
            uint32_t next = (localIndex == (PATH_PER_RAY - 1)) ? PathGuidingNode::InvalidIndex : localIndex + 1;

            // Directly send to a single location
            Vector3f radiance(100.0f);

            PathGuidingNode p;
            p.worldPosition = static_cast<float>(localIndex) * direction;
            p.prevNext = Vector<2, PathGuidingNode::IndexType>(prev, next);
            p.totalRadiance = radiance;
            // Unnecessary Data for this operation
            p.nearestDTreeIndex = DTREE_ID;
            p.radFactor = Zero3;
            paths[i] = p;
        }
        // Copy Vertices to the GPU
        CUDA_CHECK(cudaMemcpy(dPathNodes, paths.data(),
                              PATH_PER_ITERATION * sizeof(PathGuidingNode),
                              cudaMemcpyHostToDevice));
        // Do add radiance kernel
        testTree.AddRadiancesFromPaths(dPathNodes,
                                       PATH_PER_ITERATION,
                                       PATH_PER_RAY,
                                       system);
        system.SyncAllGPUs();

        // Swap the tree
        testTree.SwapTrees(FLUX_RATIO, DEPTH_LIMIT, system);
        system.SyncAllGPUs();
    }

    //// DEBUG
    //testTree.GetReadTreeToCPU(treeGPU, nodes);
    //Debug::DumpMemToFile("RT", &treeGPU, 1);
    //Debug::DumpMemToFile("RTN", nodes.data(), nodes.size());
    //testTree.GetWriteTreeToCPU(treeGPU, nodes);
    //Debug::DumpMemToFile("WT", &treeGPU, 1);
    //Debug::DumpMemToFile("WTN", nodes.data(), nodes.size());

    // Now we have very deep tree
    // Send data with zero radiance
    for(size_t i = 0; i < PATH_PER_ITERATION; i++)
    {
        uint32_t localIndex = i % PATH_PER_RAY;
        uint32_t prev = (localIndex == 0) ? PathGuidingNode::InvalidIndex : localIndex - 1;
        uint32_t next = (localIndex == (PATH_PER_RAY - 1)) ? PathGuidingNode::InvalidIndex : localIndex + 1;

        Vector3f worldUniform(uniformDist(rng), uniformDist(rng), uniformDist(rng));
        // Arbitrarily send since it is not important
        PathGuidingNode p;
        p.worldPosition = p.worldPosition = MIN_WORLD_BOUND + worldUniform * worldBound;
        p.prevNext = Vector<2, PathGuidingNode::IndexType>(prev, next);
        p.totalRadiance = Zero3;
        // Unnecessary Data for this operation
        p.nearestDTreeIndex = DTREE_ID;
        p.radFactor = Zero3;
        paths[i] = p;
    }
    // Copy Vertices to the GPU
    CUDA_CHECK(cudaMemcpy(dPathNodes, paths.data(),
                          PATH_PER_ITERATION * sizeof(PathGuidingNode),
                          cudaMemcpyHostToDevice));
    // Do add radiance kernel
    testTree.AddRadiancesFromPaths(dPathNodes,
                                   PATH_PER_ITERATION,
                                   PATH_PER_RAY, system);
    system.SyncAllGPUs();
    // Swap tree
    testTree.SwapTrees(FLUX_RATIO, DEPTH_LIMIT, system);
    system.SyncAllGPUs();

    // Now check the new trees (read and write)
    // Check integrity of the new write tree
    testTree.GetWriteTreeToCPU(treeGPU, nodes, 0);
    for(size_t i = 0; i < nodes.size(); i++)
    {
        const DTreeNode& node = nodes[i];
        EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[0]);
        EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[1]);
        EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[2]);
        EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[3]);

        if(node.parentIndex == std::numeric_limits<uint32_t>::max())
        {
            // This is root
            // Root should be the very first element
            EXPECT_EQ(0, i);
            EXPECT_EQ(treeGPU.totalSamples, 0);
            EXPECT_EQ(0.0f, treeGPU.irradiance);
            continue;
        }

        // Try to go to the parent
        const DTreeNode* n = &node;
        while(n->parentIndex != std::numeric_limits<uint32_t>::max())
        {
            n = &nodes[n->parentIndex];
        }
        // After back propogation
        // check if we actually reached to the parent
        ptrdiff_t index = n - nodes.data();
        EXPECT_EQ(0, index);
    }
    //// DEBUG
    //Debug::DumpMemToFile("WT", &treeGPU, 1);
    //Debug::DumpMemToFile("WTN", nodes.data(), nodes.size());

    // Check integrity of the new read tree
    testTree.GetReadTreeToCPU(treeGPU, nodes, 0);
    for(size_t i = 0; i < nodes.size(); i++)
    {
        const DTreeNode& node = nodes[i];
        float total = node.irradianceEstimates.Sum();
        if(node.parentIndex == std::numeric_limits<uint32_t>::max())
        {
            // This is root
            // Root should be the very first element
            EXPECT_EQ(0, i);
            EXPECT_EQ(treeGPU.totalSamples, PATH_PER_ITERATION);
            EXPECT_NEAR(treeGPU.irradiance, total,
                        MathConstants::VeryLargeEpsilon);
            continue;
        }

        const DTreeNode& parent = nodes[node.parentIndex];
        uint32_t childId = UINT32_MAX;
        childId = (parent.childIndices[0] == i) ? 0 : childId;
        childId = (parent.childIndices[1] == i) ? 1 : childId;
        childId = (parent.childIndices[2] == i) ? 2 : childId;
        childId = (parent.childIndices[3] == i) ? 3 : childId;
        EXPECT_FLOAT_EQ(total, parent.irradianceEstimates[childId]);
    }
    //// DEBUG
    //Debug::DumpMemToFile("RT", &treeGPU, 1);
    //Debug::DumpMemToFile("RTN", nodes.data(), nodes.size());
}

TEST(PPG_DTree, SampleEmpty)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    //
    constexpr uint32_t SAMPLE_COUNT = 2'500'000;
    constexpr uint32_t SEED = 0;
    RNGMemory rngMem(SEED, system);

    DTreeGroup testTree;
    testTree.AllocateDefaultTrees(1, system);

    DeviceMemory directionMemory(SAMPLE_COUNT * sizeof(Vector3f));
    DeviceMemory pdfMemory(SAMPLE_COUNT * sizeof(float));

    // Just call
    const CudaGPU& gpu = system.BestGPU();
    // Sample Tree
    gpu.GridStrideKC_X(0, 0, SAMPLE_COUNT,
                       //
                       KCSampleTree,
                       //
                       static_cast<Vector3f*>(directionMemory),
                       static_cast<float*>(pdfMemory),
                       //
                       testTree.ReadTrees(),
                       rngMem.RNGData(gpu),
                       SAMPLE_COUNT);
    // Divide by pdf
    gpu.GridStrideKC_X(0, 0, SAMPLE_COUNT,
                       //
                       KCPdfDivide,
                       //
                       static_cast<Vector3f*>(directionMemory),
                       static_cast<const float*>(pdfMemory),
                       //
                       SAMPLE_COUNT);
    // Monte Carlo
    Vector3f reducedResult;
    ReduceArrayGPU<Vector3f, ReduceAdd<Vector3f>, cudaMemcpyDeviceToHost>
    (
        reducedResult,
        static_cast<const Vector3f*>(directionMemory),
        SAMPLE_COUNT, Zero3
    );
    reducedResult /= static_cast<float>(SAMPLE_COUNT);
    // On average it should be very close to zero
    // since directions should cancel each other out
    EXPECT_NEAR(reducedResult[0], 0.0f, MathConstants::VeryLargeEpsilon);
    EXPECT_NEAR(reducedResult[1], 0.0f, MathConstants::VeryLargeEpsilon);
    EXPECT_NEAR(reducedResult[2], 0.0f, MathConstants::VeryLargeEpsilon);
}

TEST(PPG_DTree, SampleDeep)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    //
    //constexpr uint32_t SAMPLE_COUNT = 2'500'000;
    constexpr uint32_t SAMPLE_COUNT = 100;
    constexpr uint32_t SEED = 0;
    RNGMemory rngMem(SEED, system);

    constexpr int ADD_ITERATION_COUNT = 50;
    constexpr int PATH_PER_ITERATION = 50;
    constexpr int RAY_COUNT = 5;
    constexpr int PATH_PER_RAY = PATH_PER_ITERATION / RAY_COUNT;

    constexpr int DTREE_ID = 0;
    constexpr Vector3f MIN_WORLD_BOUND = Vector3f(-10, -10, -10);
    constexpr Vector3f MAX_WORLD_BOUND = Vector3f(10, 10, 10);
    // DTree parameters
    constexpr uint32_t DEPTH_LIMIT = 10;
    constexpr float FLUX_RATIO = 0.005f;

    const Vector3f direction = Vector3f(1.0f).Normalize();
    const Vector3f worldBound = MAX_WORLD_BOUND - MIN_WORLD_BOUND;
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

    std::mt19937 rng;
    rng.seed(0);
    // Check buffer
    //DTreeGPU treeGPU;
    //std::vector<DTreeNode> nodes;
    // GPU Buffers
    DeviceMemory pathNodeMemory(PATH_PER_ITERATION * sizeof(PathGuidingNode));
    PathGuidingNode* dPathNodes = static_cast<PathGuidingNode*>(pathNodeMemory);

    // First create a deep tree
    DTreeGroup testTree;
    testTree.AllocateDefaultTrees(1, system);
    std::vector<PathGuidingNode> paths(PATH_PER_ITERATION);
    for(int iCount = 0; iCount < ADD_ITERATION_COUNT; iCount++)
    {
        for(size_t i = 0; i < PATH_PER_ITERATION; i++)
        {
            uint32_t localIndex = i % PATH_PER_RAY;
            uint32_t prev = (localIndex == 0) ? PathGuidingNode::InvalidIndex : localIndex - 1;
            uint32_t next = (localIndex == (PATH_PER_RAY - 1)) ? PathGuidingNode::InvalidIndex : localIndex + 1;

            // Directly send to a single location
            Vector3f radiance(100.0f);

            PathGuidingNode p;
            p.worldPosition = static_cast<float>(localIndex) * direction;
            p.prevNext = Vector<2, PathGuidingNode::IndexType>(prev, next);
            p.totalRadiance = radiance;
            // Unnecessary Data for this operation
            p.nearestDTreeIndex = DTREE_ID;
            p.radFactor = Zero3;
            paths[i] = p;
        }
        // Copy Vertices to the GPU
        CUDA_CHECK(cudaMemcpy(dPathNodes, paths.data(),
                              PATH_PER_ITERATION * sizeof(PathGuidingNode),
                              cudaMemcpyHostToDevice));
        // Do add radiance kernel
        testTree.AddRadiancesFromPaths(dPathNodes,
                                       PATH_PER_ITERATION,
                                       PATH_PER_RAY,
                                       system);
        system.SyncAllGPUs();

        // Swap the tree
        testTree.SwapTrees(FLUX_RATIO, DEPTH_LIMIT, system);
        system.SyncAllGPUs();
    }

    // Now sample buffers are quite large so remove any other gpu memory
    pathNodeMemory = DeviceMemory();

    // Sample stuff
    DeviceMemory directionMemory(SAMPLE_COUNT * sizeof(Vector3f));
    DeviceMemory pdfMemory(SAMPLE_COUNT * sizeof(float));

    //// DEBUG
    //testTree.GetReadTreeToCPU(treeGPU, nodes);
    //Debug::DumpMemToFile("RT", &treeGPU, 1);
    //Debug::DumpMemToFile("RTN", nodes.data(), nodes.size());

    const CudaGPU& gpu = system.BestGPU();
    // Sample Tree
    gpu.GridStrideKC_X(0, 0, SAMPLE_COUNT,
                       //
                       KCSampleTree,
                       //
                       static_cast<Vector3f*>(directionMemory),
                       static_cast<float*>(pdfMemory),
                       //
                       testTree.ReadTrees(),
                       rngMem.RNGData(gpu),
                       SAMPLE_COUNT);

    std::vector<Vector3f> dirCPU(SAMPLE_COUNT);
    std::vector<float> pdfCPU(SAMPLE_COUNT);

    CUDA_CHECK(cudaMemcpy(dirCPU.data(),
                          static_cast<const Vector3f*>(directionMemory),
                          SAMPLE_COUNT * sizeof(Vector3f),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pdfCPU.data(),
                          static_cast<const float*>(pdfMemory),
                          SAMPLE_COUNT * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // TODO: Should we divide here?
    // Divide by pdf
    //gpu.GridStrideKC_X(0, 0, SAMPLE_COUNT,
    //                   //
    //                   KCPdfDivide,
    //                   //
    //                   static_cast<Vector3f*>(directionMemory),
    //                   static_cast<const float*>(pdfMemory),
    //                   //
    //                   SAMPLE_COUNT);
    // Monte Carlo
    Vector3f reducedResult;
    ReduceArrayGPU<Vector3f, ReduceAdd<Vector3f>, cudaMemcpyDeviceToHost>
    (
        reducedResult,
        static_cast<const Vector3f*>(directionMemory),
        SAMPLE_COUNT, Zero3
    );
    reducedResult /= static_cast<float>(SAMPLE_COUNT);
    // On average it should be very close to the
    // initial direction that we did give
    EXPECT_NEAR(reducedResult[0], direction[0], 0.01f);
    EXPECT_NEAR(reducedResult[1], direction[1], 0.01f);
    EXPECT_NEAR(reducedResult[2], direction[2], 0.01f);
}

TEST(PPG_DTree, DirToCoord)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    const CudaGPU& gpu = system.BestGPU();

    static constexpr uint32_t INTERVAL_COUNT = 12;
    DeviceMemory dirs(sizeof(Vector3f) * INTERVAL_COUNT);
    DeviceMemory coords(sizeof(Vector2f) * INTERVAL_COUNT);

    // Generate Directions
    std::vector<Vector2f> coordsCPU(INTERVAL_COUNT);
    for(uint32_t i = 0;i < INTERVAL_COUNT; i++)
    {
        float interval = 1.0f / static_cast<float>(INTERVAL_COUNT);
        float x = interval * static_cast<float>(i);
        coordsCPU[i] = Vector2f(x, 0.75f);
    }
    CUDA_CHECK(cudaMemcpy(static_cast<Vector2f*>(coords),
                          coordsCPU.data(),
                          sizeof(Vector2f) * INTERVAL_COUNT,
                          cudaMemcpyHostToDevice));

    // Convert it to Coords
    gpu.GridStrideKC_X(0, 0, INTERVAL_COUNT,
                       //
                       KCDirToCoord,
                       //
                       static_cast<Vector3f*>(dirs),
                       static_cast<const Vector2f*>(coords),
                       //
                       INTERVAL_COUNT);

    // Reset Coord Memory
    CUDA_CHECK(cudaMemset(static_cast<Byte*>(coords), 0x00, sizeof(Vector2f) * INTERVAL_COUNT));

    // Convert it back
    gpu.GridStrideKC_X(0, 0, INTERVAL_COUNT,
                       //
                       KCCoordToDir,
                       //
                       static_cast<Vector2f*>(coords),
                       static_cast<const Vector3f*>(dirs),
                       //
                       INTERVAL_COUNT);

    // Get to the CPU & Check
    coordsCPU = std::vector<Vector2f>(INTERVAL_COUNT);
    CUDA_CHECK(cudaMemcpy(coordsCPU.data(),
                          static_cast<Vector2f*>(coords),
                          sizeof(Vector2f) * INTERVAL_COUNT,
                          cudaMemcpyDeviceToHost));

    for(uint32_t i = 0; i < INTERVAL_COUNT; i++)
    {
        float interval = 1.0f / static_cast<float>(INTERVAL_COUNT);
        float x = interval * static_cast<float>(i);

        EXPECT_FLOAT_EQ(x, coordsCPU[i][0]);
    }
    //// DEBUG
    //std::vector<Vector3f> outDirsCPU(INTERVAL_COUNT);
    //CUDA_CHECK(cudaMemcpy(outDirsCPU.data(),
    //                      static_cast<Vector3f*>(dirs),
    //                      sizeof(Vector3f) * INTERVAL_COUNT,
    //                      cudaMemcpyDeviceToHost));
    //for(uint32_t i = 0; i < INTERVAL_COUNT; i++)
    //{
    //    printf("%f %f\n", outDirsCPU[i][0], outDirsCPU[i][2]);
    //}
}
