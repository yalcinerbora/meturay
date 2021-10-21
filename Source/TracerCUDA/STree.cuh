#pragma once

/**

STree Implementation from PPG

[Müller et al. 2017] https://dl.acm.org/doi/10.1111/cgf.13227

It is a middle-split, alternating axis binary tree. At leafs
it holds a D-tree.

This is a GPU-oriented implementation of such tree

*/

#include "DeviceMemory.h"
#include "DTree.cuh"

#include "RayLib/Types.h"
#include "RayLib/AABB.h"

class CudaSystem;

struct STreeGPU;
struct STreeNode;

class STree
{
    private:
        static constexpr size_t INITIAL_NODE_CAPACITY = 1'000;
        static constexpr size_t INITIAL_TREE_RESERVE_COUNT = 10'000;

        // Device Memory
        DeviceMemory        memory;
        STreeGPU*           dSTree;
        size_t              nodeCount;

        // DTree Allocations
        DTreeGroup          dTrees;

        void                ExpandTree(size_t newNodeCount);
        void                SplitLeaves(uint32_t maxSamplesPerNode,
                                        const CudaSystem&);
        void                SwapTrees(float fluxRatio,
                                      uint32_t depthLimit,
                                      const CudaSystem& system);
    protected:

    public:
        // Constructors & Destructor
                            STree(const AABB3f& sceneExtents,
                                  const CudaSystem& system);
                            STree(const STree&) = delete;
                            STree(STree&&) = default;
        STree&              operator=(const STree&) = delete;
        STree&              operator=(STree&&) = default;
                            ~STree() = default;

        // Members
        void                SplitAndSwapTrees(uint32_t sTreeMaxSamplePerLeaf,
                                              float dTreeFluxRatio,
                                              uint32_t dTreeDepthLimit,
                                              const CudaSystem& system);

        void                AccumulateRaidances(const PathGuidingNode* dPGNodes,
                                                uint32_t totalNodeCount,
                                                uint32_t maxPathNodePerRay,
                                                const CudaSystem&);


        void                TreeGPU(const STreeGPU*& dSTreeOut,
                                    const DTreeGPU*& dReadDTreesOut,
                                    DTreeGPU*& dWriteDTreesOut);
        uint32_t            TotalTreeCount() const;

        size_t              UsedGPUMemory() const;
        size_t              UsedCPUMemory() const;

        // Debugging
        void                GetTreeToCPU(STreeGPU&, std::vector<STreeNode>&) const;
        void                GetAllDTreesToCPU(std::vector<DTreeGPU>&,
                                              std::vector<std::vector<DTreeNode>>&,
                                              bool fetchReadTree) const;

        void                DumpSDTreeAsBinary(std::vector<Byte>& data,
                                               bool fetchReadTree) const;

};

inline void STree::TreeGPU(const STreeGPU*& dSTreeOut,
                           const DTreeGPU*& dReadDTreesOut,
                           DTreeGPU*& dWriteDTreesOut)
{
    dSTreeOut = dSTree;
    dReadDTreesOut = dTrees.ReadTrees();
    dWriteDTreesOut = dTrees.WriteTrees();
}

inline uint32_t STree::TotalTreeCount() const
{
    return dTrees.TreeCount();
}

inline size_t STree::UsedGPUMemory() const
{
    size_t total = dTrees.UsedGPUMemory() + memory.Size();
    return total;
}

inline size_t STree::UsedCPUMemory() const
{
    size_t total = dTrees.UsedCPUMemory();
    return total;
}