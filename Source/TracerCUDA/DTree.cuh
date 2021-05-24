﻿#pragma once

/**

DTree Implementation

*/ 

#include "DeviceMemory.h"
#include <vector>

struct DTreeNode;
struct DTreeGPU;
struct PathNode;

class CudaGPU;

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


                DTreeGPU*           TreeGPU();
                const DTreeGPU*     TreeGPU() const;

                size_t              NodeCount() const;
                void                ResetAndReserve(size_t nodeCount,
                                                    const CudaGPU&,
                                                    cudaStream_t);
                void                CopyGPUNodeCountToCPU();

                void                DumpTree(DTreeGPU&, std::vector<DTreeNode>&) const;
        };

    private:
        // Tree Nodes (one for read other for write
        DTreeBuffer            readTree;
        DTreeBuffer            writeTree;

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
        void                    SwapTrees(const CudaGPU&, float fluxRatio, 
                                          uint32_t depthLimit);
        
        void                    GetReadTreeToCPU(DTreeGPU&, std::vector<DTreeNode>&) const;
        void                    GetWriteTreeToCPU(DTreeGPU&, std::vector<DTreeNode>&) const;

};

inline DTreeGPU* DTree::DTreeBuffer::TreeGPU()
{
    return dDTree;
}
inline const DTreeGPU* DTree::DTreeBuffer::TreeGPU() const
{
    return dDTree;
}

inline size_t DTree::DTreeBuffer::NodeCount() const
{
    return nodeCount;
}