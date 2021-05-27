#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <numeric>
#include <random>

#include "TracerCUDA/STree.cuh"
#include "TracerCUDA/STreeKC.cuh"

#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/TracerDebug.h"

TEST(PPG_STree, Empty)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    //// Constants
    //// If a node has %10 or more total energy, split
    //static constexpr float FLUX_RATIO = 0.1f;
    //// Maximum allowed depth of the tree
    //static constexpr uint32_t DEPTH_LIMIT = 10;

    //std::vector<STreeNode> nodes;
    //STreeGPU tree;
}