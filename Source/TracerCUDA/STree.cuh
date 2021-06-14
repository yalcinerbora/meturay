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

#include <RayLib/AABB.h>

class CudaSystem;

struct STreeGPU;
struct STreeNode;

class STree
{
    public:
        static constexpr uint32_t InvalidDTreeIndex = std::numeric_limits<uint32_t>::max();

    private:
        static constexpr size_t INITIAL_NODE_CAPACITY = 1'000;
        static constexpr size_t INITIAL_TREE_RESERVE_COUNT = 10'000;

        // Device Memory
        DeviceMemory        memory;
        STreeGPU*           dSTree;
        size_t              nodeCount;
       
        // DTree Allocations
        std::vector<DTree>  dTrees;

        // DTree Buffer for Tracer
        DeviceMemory        rwDTreeGPUBuffer;
        const DTreeGPU**    dReadDTrees;
        DTreeGPU**          dWriteDTrees;
        
        void                LinearizeDTreeGPUPtrs(DeviceMemory&,
                                                  bool readTree, size_t offset = 0);
        void                LinearizeDTreeGPUPtrs(DeviceMemory&);
        void                ExpandTree(size_t newNodeCount);
        void                SplitLeaves(uint32_t maxSamplesPerNode,
                                        const CudaSystem&);
        void                SwapTrees(float fluxRatio,
                                      uint32_t depthLimit,
                                      const CudaSystem& system);
    protected:

    public:
        // Constructors & Destructor
                            STree(const AABB3f& sceneExtents);
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


        void                TreeGPU(const STreeGPU*& dSTree,
                                    const DTreeGPU**& dReadDTrees,
                                    DTreeGPU**& dWriteDTrees) const;
        uint32_t            TotalTreeCount() const;

        size_t              UsedGPUMemory() const;
        size_t              UsedCPUMemory() const;

        // Debugging
        void                GetTreeToCPU(STreeGPU&, std::vector<STreeNode>&) const;
        void                GetAllDTreesToCPU(std::vector<DTreeGPU>&,
                                              std::vector<std::vector<DTreeNode>>&,
                                              bool fetchReadTree) const;

};

inline void STree::TreeGPU(const STreeGPU*& dSTreeOut,
                           const DTreeGPU**& dReadDTreesOut,
                           DTreeGPU**& dWriteDTreesOut) const
{
    dSTreeOut = dSTree;
    dReadDTreesOut = dReadDTrees;
    dWriteDTreesOut = dWriteDTrees;
}

inline uint32_t STree::TotalTreeCount() const
{
    return static_cast<uint32_t>(dTrees.size());
}

inline size_t STree::UsedGPUMemory() const
{
    size_t total = memory.Size();
    for(const DTree& dt : dTrees)
        total += dt.UsedGPUMemory();
    return total;
}

inline size_t STree::UsedCPUMemory() const
{
    size_t total = 0;
    for(const DTree& dt : dTrees)
        total += dt.UsedCPUMemory();
    return total;
}