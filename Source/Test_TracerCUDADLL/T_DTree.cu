#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "TracerCUDA/DTree.cuh"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/DTreeKC.cuh"
#include "TracerCUDA/PathNode.h"

using ::testing::FloatEq;

TEST(PPG_DTree, Empty)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    // Constants
    // If a node has %10 or more total energy, split
    static constexpr float fluxRatio = 0.1f;
    // Maximum allowed depth of the tree
    static constexpr uint32_t depthLimit = 10;

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
    testTree.SwapTrees(system.BestGPU(), fluxRatio, depthLimit);
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
    static constexpr float fluxRatio = 0.1f;
    // Maximum allowed depth of the tree
    static constexpr uint32_t depthLimit = 10;

    std::vector<DTreeNode> nodes;
    DTreeGPU tree;

    PathGuidingNode cd = PathGuidingNode
    {
        PathNode
        {
            Vector3f{0.0f, 0.0f, 0.0f},
            Vector<2, PathNode::IndexType>{PathNode::InvalidIndex, 1}
        },
        Vector3f{1.0f, 1.0f, 1.0f},
        static_cast<uint32_t>(0u),
        Vector3f{10.0f, 10.0f, 10.0f}
    };

    std::vector<PathGuidingNode> pathNodes =
    {
        PathGuidingNode
        {
            PathNode
            {
                Vector3f{0.0f, 0.0f, 0.0f},
                Vector<2, PathNode::IndexType>{PathNode::InvalidIndex, 1}
            },
            Vector3f{1.0f, 1.0f, 1.0f},
            static_cast<uint32_t>(0u),
            Vector3f{10.0f, 10.0f, 10.0f}
        },
        PathGuidingNode
        {
            PathNode
            {
                Vector3f{2.0f, 4.0f, 5.0f},
                Vector<2, PathNode::IndexType>{0, 2}
            },
            Vector3f{0.8f, 0.8f, 0.8f},
            static_cast<uint32_t>(0u),
            Vector3f{70.0f, 2.0f, 33.0f}
        },
        PathGuidingNode
        {
            PathNode
            {
                Vector3f{-3.0f, -5.0f, -2.0f},
                Vector<2, PathNode::IndexType>{1, PathNode::InvalidIndex}
            },
            Vector3f{0.5f, 0.5f, 0.5f},
            static_cast<uint32_t>(0u),
            Vector3f{1.0f, 2.0f, 3.0f}
        },
    };

    std::vector<Vector3f> directions;
    for(const PathGuidingNode& p : pathNodes)
    {
        if(p.prevNext[1] != PathNode::InvalidIndex)
        {
            directions.push_back(pathNodes[p.prevNext[0]].worldPosition - p.worldPosition);
            directions.back().NormalizeSelf();
        }            
    }

    // Copy Vertices to the GPU
    
}