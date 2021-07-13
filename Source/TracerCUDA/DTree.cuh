#pragma once

/**

STree Implementation from PPG

[Müller et al. 2017] https://dl.acm.org/doi/10.1111/cgf.13227

*/ 

#include "DeviceMemory.h"

#include "RayLib/ArrayPortion.h"
#include "RayLib/Types.h"

#include <vector>

struct DTreeNode;
struct DTreeGPU;
struct PathGuidingNode;

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
                void                CopyGPUNodeCountToCPU(cudaStream_t stream);
                size_t              UsedGPUMemory() const;

                void                DumpTree(DTreeGPU&, std::vector<DTreeNode>&) const;
                void                DumpTreeAsBinary(std::vector<Byte>&) const;
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
        void                    SwapTrees(float fluxRatio, 
                                          uint32_t depthLimit,
                                          const CudaGPU&);
        void                    AddRadiancesFromPaths(const uint32_t* dNodeIndexArray,
                                                      const PathGuidingNode* dPathNodes,
                                                      const ArrayPortion<uint32_t>& portion,
                                                      uint32_t maxPathNodePerRay,
                                                      const CudaGPU&);
        // Access
        DTreeGPU*               TreeGPU(bool readTree);
        const DTreeGPU*         TreeGPU(bool writeTree) const;

        size_t                  UsedGPUMemory() const;
        size_t                  UsedCPUMemory() const;

        size_t                  NodeCount(bool readOrWriteTree) const;

        // Debugging
        void                    GetReadTreeToCPU(DTreeGPU&, std::vector<DTreeNode>&) const;
        void                    GetWriteTreeToCPU(DTreeGPU&, std::vector<DTreeNode>&) const;
        void                    DumpTreeAsBinary(std::vector<Byte>&, bool fetchReadTree) const;

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

inline size_t DTree::DTreeBuffer::UsedGPUMemory() const
{
    return memory.Size();
}

inline DTreeGPU* DTree::TreeGPU(bool fetchReadTree)
{
    DTreeBuffer& b = (fetchReadTree) ? readTree : writeTree;
    return b.TreeGPU();
}

inline const DTreeGPU* DTree::TreeGPU(bool fetchReadTree) const
{
    const DTreeBuffer& b = (fetchReadTree) ? readTree : writeTree;
    return b.TreeGPU();
}

inline size_t DTree::UsedGPUMemory() const
{
    return (writeTree.UsedGPUMemory() +
            readTree.UsedGPUMemory());
}

inline size_t DTree::UsedCPUMemory() const
{
    return sizeof(DTree);
}

inline size_t DTree::NodeCount(bool readOrWriteTree) const
{
    if(readOrWriteTree)
        return readTree.NodeCount();
    else
        return writeTree.NodeCount();
}