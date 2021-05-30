﻿#pragma once

/**

STree Implementation from PPG 

[Müller et al. 2017] https://dl.acm.org/doi/10.1111/cgf.13227

It is a middle-split, alternating axis binary tree. At leafs
it holds a D-tree.

This is a GPU-oriented implementation of such tree

*/ 

#include "STreeKC.cuh"
#include "DeviceMemory.h"
#include "DTree.cuh"

class CudaSystem;

class STree
{
    private:
        static constexpr size_t INITIAL_NODE_CAPACITY = 1'000;

        // Device Memory
        DeviceMemory        memory;
        STreeGPU*           dSTree;
        size_t              nodeCount;
       
        // DTree Allocations
        std::vector<DTree>  dTrees;

        // DTree Buffer for Tracer
        DeviceMemory        readDTreeGPUBuffer;
        const DTreeGPU**    dReadDTrees;
        
        DeviceMemory        LinearizeDTreeGPUPtrs(bool readTree);
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
                                    const DTreeGPU**& dDTrees) const;
        uint32_t            TotalTreeCount() const;

        // Debugging
        void                GetTreeToCPU(STreeGPU&, std::vector<STreeNode>&) const;


};

inline void STree::TreeGPU(const STreeGPU*& dSTreeOut,
                           const DTreeGPU**& dDTreesOut) const
{
    dSTreeOut = dSTree;
    dDTreesOut = dReadDTrees;
}

inline uint32_t STree::TotalTreeCount() const
{
    return static_cast<uint32_t>(dTrees.size());
}