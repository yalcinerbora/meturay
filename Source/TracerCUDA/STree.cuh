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
        DeviceMemory        sTreeNodeMemory;
        STreeNode*          dSTreeNodes;
        //
        STreeGPU            sTreeGPU;

        std::vector<DTree>  dTrees;
        
        

    protected:

    public:
        // Constructors & Destructor
                        STree(const AABB3f& sceneExtents);
                        STree(const STree&) = delete;
                        STree(STree&&) = default;
        STree&          operator=(const STree&) = delete;
        STree&          operator=(STree&&) = default;
                        ~STree() = default;

        // Members
        void            SplitLeaves(const CudaSystem&);


};