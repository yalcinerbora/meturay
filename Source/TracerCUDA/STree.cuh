#pragma once

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
        const DTreeGPU*     dDTreesRead;
        
        void                ExpandTree(size_t newNodeCount);

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
        void                SplitLeaves(const CudaSystem&);
        void                AccumulateRaidances(const PathGuidingNode* dPGNodes, 
                                                uint32_t totalNodeCount,
                                                uint32_t maxPathNodePerRay,
                                                const CudaSystem&);

        void                TreeGPU(const STreeGPU*& dSTree,
                                    const DTreeGPU*& dDTrees) const;


};

inline void STree::TreeGPU(const STreeGPU*& dSTreeOut,
                           const DTreeGPU*& dDTreesOut) const
{
    dSTreeOut = dSTree;
    dDTreesOut = dDTreesRead;
}