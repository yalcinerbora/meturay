#pragma once

/**

DTree Implementation

*/ 

#include "RayLib/Vector.h"

#include "DeviceMemory.h"

struct DTreeNode
{
    enum NodeOrder
    {
        BOTTOM_LEFT     = 0,
        BOTTOM_RIGHT    = 1,
        TOP_LEFT        = 2,
        TOP_RIGHT       = 3
    };

    Vector4ui   childIndex;
    Vector4f    cdfs;
};

class DTree
{
    private:
        DeviceMemory    treeNodeMemory;
        DTreeNode*      dTreeNodes;

    protected:

    public:
        // Constructors & Destructor
                        DTree();
                        DTree(const DTree&) = delete;
                        DTree(DTree&&) = default;
        DTree&          operator=(const DTree&) = delete;
        DTree&          operator=(DTree&&) = default;
                        ~DTree() = default;

        // Members
        
};