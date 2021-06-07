#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <numeric>
#include <random>

#include "TracerCUDA/DTree.cuh"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/DTreeKC.cuh"

#include "TracerCUDA/TracerDebug.h"

using ::testing::FloatEq;

static std::ostream& operator<<(std::ostream& s, const PathGuidingNode& n)
{
    s << "{"   << std::endl
      << "   " << n.worldPosition[0] << ", " 
               << n.worldPosition[1] << ", "
               << n.worldPosition[2] << std::endl
      << "   " << static_cast<uint32_t>(n.prevNext[0]) << ", "
               << static_cast<uint32_t>(n.prevNext[1]) << std::endl
      << "}";
    return s;
}

static std::ostream& operator<<(std::ostream& s, const DTreeNode& n)
{
    constexpr uint32_t UINT32_T_MAX = std::numeric_limits<uint32_t>::max();
    constexpr uint16_t UINT16_T_MAX = std::numeric_limits<uint16_t>::max();

    s << "P{"; 
    if(n.parentIndex == UINT16_T_MAX) s << "-";
    else s << n.parentIndex;
    s << "} ";
    s << "C{";
    if(n.childIndices[0] == UINT32_T_MAX) s << "-";
    else s << n.childIndices[0];
    s << ", ";
    if(n.childIndices[1] == UINT32_T_MAX) s << "-";
    else s << n.childIndices[1];
    s << ", ";
    if(n.childIndices[2] == UINT32_T_MAX) s << "-";
    else s << n.childIndices[2];
    s << ", ";
    if(n.childIndices[3] == UINT32_T_MAX) s << "-";
    else s << n.childIndices[3];
    s << "} ";
    s << "I{"
      << n.irradianceEstimates[0] << ", "
      << n.irradianceEstimates[1] << ", "
      << n.irradianceEstimates[2] << ", "
      << n.irradianceEstimates[3] << "}";
    return s;
}

static std::ostream& operator<<(std::ostream& s, const DTreeGPU& n)
{
    s << "Irradiane  : " << n.irradiance << std::endl;
    s << "NodeCount  : " << n.nodeCount << std::endl;
    s << "SampleCount: " << n.totalSamples << std::endl;        
    return s;
}

TEST(PPG_DTree, Empty)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    // Constants
    // If a node has %10 or more total energy, split
    static constexpr float FLUX_RATIO = 0.1f;
    // Maximum allowed depth of the tree
    static constexpr uint32_t DEPTH_LIMIT = 10;

    std::vector<DTreeNode> nodes;
    DTreeGPU tree;

    // Initialize Check
    DTree testTree;
    testTree.GetReadTreeToCPU(tree, nodes);
    EXPECT_EQ(0.0f, tree.irradiance);
    EXPECT_EQ(0, tree.totalSamples);
    EXPECT_EQ(1, tree.nodeCount);
    EXPECT_EQ(1, nodes.size());
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[0]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[1]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[2]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[3]);
    EXPECT_EQ(std::numeric_limits<uint16_t>::max(), nodes.front().parentIndex);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[0]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[1]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[2]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[3]);
    testTree.GetWriteTreeToCPU(tree, nodes);
    EXPECT_EQ(0.0f, tree.irradiance);
    EXPECT_EQ(0, tree.totalSamples);
    EXPECT_EQ(1, tree.nodeCount);
    EXPECT_EQ(1, nodes.size());
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[0]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[1]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[2]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[3]);
    EXPECT_EQ(std::numeric_limits<uint16_t>::max(), nodes.front().parentIndex);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[0]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[1]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[2]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[3]);

    // After Swap Check
    testTree.SwapTrees(FLUX_RATIO, DEPTH_LIMIT, system.BestGPU());
    system.SyncAllGPUs();
    testTree.GetReadTreeToCPU(tree, nodes);
    EXPECT_EQ(0.0f, tree.irradiance);
    EXPECT_EQ(0, tree.totalSamples);
    EXPECT_EQ(1, tree.nodeCount);
    EXPECT_EQ(1, nodes.size());
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[0]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[1]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[2]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[3]);
    EXPECT_EQ(std::numeric_limits<uint16_t>::max(), nodes.front().parentIndex);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[0]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[1]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[2]);
    EXPECT_EQ(0.0f, nodes.front().irradianceEstimates[3]);
    testTree.GetWriteTreeToCPU(tree, nodes);
    EXPECT_EQ(0.0f, tree.irradiance);
    EXPECT_EQ(0, tree.totalSamples);
    EXPECT_EQ(1, tree.nodeCount);
    EXPECT_EQ(1, nodes.size());
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[0]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[1]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[2]);
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), nodes.front().childIndices[3]);
    EXPECT_EQ(std::numeric_limits<uint16_t>::max(), nodes.front().parentIndex);
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

    // Check buffers
    std::vector<DTreeNode> nodes;
    DTreeGPU treeGPU;

    PathGuidingNode camNode;
    camNode.worldPosition = Vector3f{0.0f, 0.0f, 0.0f};
    camNode.prevNext = Vector<2, PathNode::IndexType>(PathNode::InvalidIndex, 1);
    camNode.radFactor = Zero3;
    camNode.nearestDTreeIndex = 0;
    camNode.totalRadiance = Vector3f{10.0f, 10.0f, 10.0f};
    PathGuidingNode midNode0;
    midNode0.worldPosition = Vector3f{1.0f, 1.0f, 1.0f};
    midNode0.prevNext = Vector<2, PathNode::IndexType>(0, 2);
    midNode0.radFactor = Zero3;
    midNode0.nearestDTreeIndex = 0;
    midNode0.totalRadiance = Vector3f{30.0f, 30.0f, 30.0f};
    PathGuidingNode midNode1;
    midNode1.worldPosition = Vector3f{0.0f, 0.0f, 0.0f};
    midNode1.prevNext = Vector<2, PathNode::IndexType>(1, 3);
    midNode1.radFactor = Zero3;
    midNode1.nearestDTreeIndex = 0;
    midNode1.totalRadiance = Vector3f{40.0f, 40.0f, 40.0f};
    PathGuidingNode midNode2;
    midNode2.worldPosition = Vector3f{1.0f, 1.0f, -1.0f};
    midNode2.prevNext = Vector<2, PathNode::IndexType>(2, 4);
    midNode2.radFactor = Zero3;
    midNode2.nearestDTreeIndex = 0;
    midNode2.totalRadiance = Vector3f{50.0f, 50.0f, 50.0f};
    PathGuidingNode endNode;
    endNode.worldPosition = Vector3f{0.0f, 0.0f, 0.0f};
    endNode.prevNext = Vector<2, PathNode::IndexType>(3, PathNode::InvalidIndex);
    endNode.radFactor = Zero3;
    endNode.nearestDTreeIndex = 0;
    endNode.totalRadiance = Vector3f{0.0f, 0.0f, 0.0f};

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
    DTree testTree;
    // Copy Vertices to the GPU
    DeviceMemory pathNodeMemory(pathNodes.size() * sizeof(PathGuidingNode));
    PathGuidingNode* dPathNodes = static_cast<PathGuidingNode*>(pathNodeMemory);
    CUDA_CHECK(cudaMemcpy(dPathNodes, pathNodes.data(),
                          pathNodes.size() * sizeof(PathGuidingNode),
                          cudaMemcpyHostToDevice));

    DeviceMemory indexMemory(pathNodes.size() * sizeof(uint32_t));
    uint32_t* dIndices = static_cast<uint32_t*>(indexMemory);
    std::vector<uint32_t> hIndices(pathNodes.size());
    std::iota(hIndices.begin(), hIndices.end(), 0);
    CUDA_CHECK(cudaMemcpy(dIndices, hIndices.data(),
                          pathNodes.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Push these values to the Tree
    const uint32_t PathNodePerRay = static_cast<uint32_t>(pathNodes.size());
    testTree.AddRadiancesFromPaths(dIndices, dPathNodes,
                                   ArrayPortion<uint32_t>{0, 0, pathNodes.size()},
                                   PathNodePerRay, system.BestGPU());
    system.SyncAllGPUs();

    // Check Tree
    testTree.GetWriteTreeToCPU(treeGPU, nodes);
    for(size_t i = 0; i < nodes.size(); i++)
    {
        const DTreeNode& node = nodes[i];
        if(node.parentIndex == std::numeric_limits<uint16_t>::max())
        {
            // This is root
            // Root should be the very first element
            EXPECT_EQ(0, i);
            EXPECT_EQ(treeGPU.totalSamples, pathNodes.size());
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

    // Do the swap
    testTree.SwapTrees(FLUX_RATIO, DEPTH_LIMIT, system.BestGPU());
    system.SyncAllGPUs();

    // Check again
    testTree.GetReadTreeToCPU(treeGPU, nodes);
    for(size_t i = 0; i < nodes.size(); i++)
    {
        const DTreeNode& node = nodes[i];
        float total = node.irradianceEstimates.Sum();
        if(node.parentIndex == std::numeric_limits<uint16_t>::max())
        {
            // This is root
            // Root should be the very first element
            EXPECT_EQ(0, i);
            EXPECT_EQ(treeGPU.totalSamples, pathNodes.size());
            EXPECT_FLOAT_EQ(treeGPU.irradiance, total);
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
    testTree.GetWriteTreeToCPU(treeGPU, nodes);
    for(size_t i = 0; i < nodes.size(); i++)
    {
        const DTreeNode& node = nodes[i];
        float total = node.irradianceEstimates.Sum();
        if(node.parentIndex == std::numeric_limits<uint16_t>::max())
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

TEST(PPG_DTree, Stress)
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
    DeviceMemory indexMemory(PATH_PER_ITERATION * sizeof(uint32_t));
    uint32_t* dIndices = static_cast<uint32_t*>(indexMemory);

    // Copy redundant incrementing buffer to GPU (since we are not sorting stuff)
    std::vector<uint32_t> hIndices(PATH_PER_ITERATION);
    std::iota(hIndices.begin(), hIndices.end(), 0);
    CUDA_CHECK(cudaMemcpy(dIndices, hIndices.data(),
                          PATH_PER_ITERATION * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    hIndices.clear();
    
    // Check buffer
    DTreeGPU treeGPU;
    std::vector<DTreeNode> nodes;

    // Stress the Tree by randomly adding data multiple times
    DTree testTree;
    std::vector<PathGuidingNode> paths(PATH_PER_ITERATION);
    for(int iCount = 0; iCount < ITERATION_COUNT; iCount++)
    {
        // Constants for this itertion
        // If a node has %X or more total energy, split
        const float fluxRatio = FLUX_MIN + uniformDist(rng) * (FLUX_MAX - FLUX_MIN);
        // Maximum allowed depth of the tree
        uint32_t depthLimit = DEPTH_MIN + static_cast<uint32_t>(uniformDist(rng) * (DEPTH_MAX - DEPTH_MIN));

        //METU_LOG("Depth %u, Flux %f", depthLimit, fluxRatio);

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


        // Copy Vertices to the GPU
        CUDA_CHECK(cudaMemcpy(dPathNodes, paths.data(),
                              PATH_PER_ITERATION * sizeof(PathGuidingNode),
                              cudaMemcpyHostToDevice));
        // Do add radiance kernel
        testTree.AddRadiancesFromPaths(dIndices, dPathNodes,
                                       ArrayPortion<uint32_t>{DTREE_ID, 0, PATH_PER_ITERATION},
                                       PATH_PER_RAY, system.BestGPU());
        system.SyncAllGPUs();

        // Check if radiance is properly added
        testTree.GetWriteTreeToCPU(treeGPU, nodes);
        for(size_t i = 0; i < nodes.size(); i++)
        {
            const DTreeNode& node = nodes[i];
            if(node.parentIndex == std::numeric_limits<uint16_t>::max())
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

        testTree.SwapTrees(fluxRatio, depthLimit, system.BestGPU());
        system.SyncAllGPUs();

        // Check integrity of the new write tree
        testTree.GetWriteTreeToCPU(treeGPU, nodes);
        for(size_t i = 0; i < nodes.size(); i++)
        {
            const DTreeNode& node = nodes[i];
            EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[0]);
            EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[1]);
            EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[2]);
            EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[3]);

            if(node.parentIndex == std::numeric_limits<uint16_t>::max())
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
            while(n->parentIndex != std::numeric_limits<uint16_t>::max())
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
        testTree.GetReadTreeToCPU(treeGPU, nodes);
        for(size_t i = 0; i < nodes.size(); i++)
        {
            const DTreeNode& node = nodes[i];
            float total = node.irradianceEstimates.Sum();
            if(node.parentIndex == std::numeric_limits<uint16_t>::max())
            {
                // This is root
                // Root should be the very first element
                EXPECT_EQ(0, i);
                EXPECT_EQ(treeGPU.totalSamples, PATH_PER_ITERATION);
                EXPECT_FLOAT_EQ(treeGPU.irradiance, total);
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

TEST(PPG_DTree, ZeroAdd)
{
        CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    constexpr int ITERATION_COUNT = 500;
    constexpr int PATH_PER_ITERATION = 5000;
    constexpr int RAY_COUNT = 500;
    constexpr int PATH_PER_RAY = PATH_PER_ITERATION / RAY_COUNT;

    constexpr int DTREE_ID = 0;
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
    DeviceMemory indexMemory(PATH_PER_ITERATION * sizeof(uint32_t));
    uint32_t* dIndices = static_cast<uint32_t*>(indexMemory);

    // Copy redundant incrementing buffer to GPU (since we are not sorting stuff)
    std::vector<uint32_t> hIndices(PATH_PER_ITERATION);
    std::iota(hIndices.begin(), hIndices.end(), 0);
    CUDA_CHECK(cudaMemcpy(dIndices, hIndices.data(),
                          PATH_PER_ITERATION * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    hIndices.clear();
    
    // Check buffer
    DTreeGPU treeGPU;
    std::vector<DTreeNode> nodes;

    // Stress the Tree by randomly adding data multiple times
    DTree testTree;
    std::vector<PathGuidingNode> paths(PATH_PER_ITERATION);
    for(int iCount = 0; iCount < ITERATION_COUNT; iCount++)
    {
        // Constants for this itertion
        // If a node has %X or more total energy, split
        const float fluxRatio = FLUX_MIN + uniformDist(rng) * (FLUX_MAX - FLUX_MIN);
        // Maximum allowed depth of the tree
        uint32_t depthLimit = DEPTH_MIN + static_cast<uint32_t>(uniformDist(rng) * (DEPTH_MAX - DEPTH_MIN));

        //METU_LOG("Depth %u, Flux %f", depthLimit, fluxRatio);

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
        testTree.AddRadiancesFromPaths(dIndices, dPathNodes,
                                       ArrayPortion<uint32_t>{DTREE_ID, 0, PATH_PER_ITERATION},
                                       PATH_PER_RAY, system.BestGPU());
        system.SyncAllGPUs();

        // Check if radiance is properly added
        testTree.GetWriteTreeToCPU(treeGPU, nodes);
        for(size_t i = 0; i < nodes.size(); i++)
        {
            const DTreeNode& node = nodes[i];
            if(node.parentIndex == std::numeric_limits<uint16_t>::max())
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

        testTree.SwapTrees(fluxRatio, depthLimit, system.BestGPU());
        system.SyncAllGPUs();

        // Check integrity of the new write tree
        testTree.GetWriteTreeToCPU(treeGPU, nodes);
        for(size_t i = 0; i < nodes.size(); i++)
        {
            const DTreeNode& node = nodes[i];
            EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[0]);
            EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[1]);
            EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[2]);
            EXPECT_FLOAT_EQ(0.0f, node.irradianceEstimates[3]);

            if(node.parentIndex == std::numeric_limits<uint16_t>::max())
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
            while(n->parentIndex != std::numeric_limits<uint16_t>::max())
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
        testTree.GetReadTreeToCPU(treeGPU, nodes);
        for(size_t i = 0; i < nodes.size(); i++)
        {
            const DTreeNode& node = nodes[i];
            float total = node.irradianceEstimates.Sum();
            if(node.parentIndex == std::numeric_limits<uint16_t>::max())
            {
                // This is root
                // Root should be the very first element
                EXPECT_EQ(0, i);
                EXPECT_EQ(treeGPU.totalSamples, PATH_PER_ITERATION);
                EXPECT_FLOAT_EQ(treeGPU.irradiance, total);
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