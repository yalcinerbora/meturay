#pragma once

/**

DTree Implementation

*/ 

#include "DeviceMemory.h"

struct DTreeNode;
struct DTreeGPU;

class CudaSystem;

class DTree
{
    public: 
        class DTreeBuffer
        {
            private:
                DeviceMemory    memory;
                DTreeGPU*       dDTree;
                size_t          nodeCount;

                void            FixPointers();

            protected:
            public:
                // Constructors & Destructor
                                    DTreeBuffer();
                                    DTreeBuffer(const DTreeBuffer&);
                                    DTreeBuffer(DTreeBuffer&&) = default;
                DTreeBuffer&        operator=(const DTreeBuffer&);
                DTreeBuffer&        operator=(DTreeBuffer&&) = default;
                                    ~DTreeBuffer() = default;
        };

    private:
        // Tree Nodes (one for read other for write
        DTreeBuffer            dReadTree;
        DTreeBuffer            dWriteTree;

    protected:
    public:
        // Constructors & Destructor
                                DTree() = default;
                                DTree(const DTree&) = default;
                                DTree(DTree&&) = default;
        DTree&                  operator=(const DTree&) = default;
        DTree&                  operator=(DTree&&) = default;
                                ~DTree() = default;

        // Members
        void                    SwapTrees(const CudaSystem&);
        
};