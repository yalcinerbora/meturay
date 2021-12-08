#pragma once

/**

STree Implementation from PPG

[Müller et al. 2017] https://dl.acm.org/doi/10.1111/cgf.13227

*/

#include "DeviceMemory.h"

#include "RayLib/ArrayPortion.h"
#include "RayLib/Types.h"
#include "RayLib/CudaCheck.h"
#include "RayLib/Constants.h"

#include <vector>
#include <cassert>

struct DTreeNode;
struct DTreeGPU;
struct PathGuidingNode;

class CudaSystem;

class DTreeGroup
{
    private:
        class DTreeBuffer
        {
            private:
                DeviceMemory            treeNodeMemory;
                DTreeNode*              dDTreeNodes;

                std::vector<uint32_t>   hDTreeNodeOffsets;

                DeviceMemory            offsetMemory;
                uint32_t*               dDTreeNodeOffsets;

                DeviceMemory            treeMemory;
                DTreeGPU*               dDTrees;

            protected:
            public:
                // Constructors & Destructor
                                        DTreeBuffer();
                                        DTreeBuffer(const DTreeBuffer&) = delete;
                                        DTreeBuffer(DTreeBuffer&&) = default;
                DTreeBuffer&            operator=(const DTreeBuffer&) = delete;
                DTreeBuffer&            operator=(DTreeBuffer&&) = default;
                                        ~DTreeBuffer() = default;

                void                    AllocateDefaultTrees(uint32_t count, bool setRootIrrad,
                                                             const CudaSystem& system);
                void                    AllocateExtra(const std::vector<uint32_t>& oldTreeIds,
                                                      const CudaSystem& system);
                void                    ResetAndReserve(const uint32_t* newNodeCounts,
                                                        uint32_t newTreeCount,
                                                        const CudaSystem& system);
                void                    InitializeTrees(const std::vector<std::pair<uint32_t, float>>& hDTreeBases,
                                                        const std::vector<std::vector<DTreeNode>>& hDTreeNodes,
                                                        bool purgeValues,
                                                        const CudaSystem&);


                uint32_t                DTreeCount() const;
                uint32_t                DTreeTotalNodeCount() const;
                uint32_t                DTreeNodeCount(uint32_t) const;
                uint32_t                DTreeNodeOffset(uint32_t) const;
                const uint32_t*         DTreeNodeOffsetsGPU() const;

                const DTreeNode*        DTreeNodesGPU() const;
                DTreeNode*              DTreeNodesGPU();

                const DTreeGPU*         DTrees() const;
                DTreeGPU*               DTrees();

                size_t                  UsedGPUMemory() const;
                size_t                  UsedCPUMemory() const;

                void                    GetTreeToCPU(DTreeGPU&, std::vector<DTreeNode>&, uint32_t treeIndex) const;
                void                    DumpTreeAsBinary(std::vector<Byte>&, uint32_t& nodeCount, uint32_t treeIndex) const;
        };

    public:
        static constexpr uint32_t       InvalidDTreeIndex = std::numeric_limits<uint32_t>::max();
        static constexpr uint32_t       InvalidParentIndex = std::numeric_limits<uint32_t>::max();
        static constexpr uint32_t       InvalidChildIndex = std::numeric_limits<uint32_t>::max();

    private:
        DTreeBuffer             readTrees;
        DTreeBuffer             writeTrees;

    protected:
    public:
        // Constructors & Destructor
                                DTreeGroup() = default;
                                DTreeGroup(const DTreeGroup&) = delete;
                                DTreeGroup(DTreeGroup&&) = delete;
        DTreeGroup&             operator=(const DTreeGroup&) = delete;
        DTreeGroup&             operator=(DTreeGroup&&) = delete;
                                ~DTreeGroup() = default;


        // Interface
        // Allocate extra trees and copy them from the old tree ids
        void                    AllocateDefaultTrees(uint32_t count, const CudaSystem& system);
        void                    AllocateExtra(const std::vector<uint32_t>& oldTreeIds,
                                              const CudaSystem& system);
        void                    SwapTrees(float fluxRatio, uint32_t depthLimit,
                                          const CudaSystem& system);
        void                    AddRadiancesFromPaths(const PathGuidingNode* dPGNodes,
                                                      uint32_t totalNodeCount,
                                                      uint32_t maxPathNodePerRay,
                                                      const CudaSystem& system);

        //
        void                    InitializeTrees(const std::vector<std::pair<uint32_t, float>>&,
                                                const std::vector<std::vector<DTreeNode>>&,
                                                const CudaSystem&);

        uint32_t                TreeCount() const;
        size_t                  NodeCount(uint32_t treeIndex, bool readOrWriteTree) const;

        DTreeGPU*               ReadTrees();
        const DTreeGPU*         ReadTrees() const;
        DTreeGPU*               WriteTrees();
        const DTreeGPU*         WriteTrees() const;
        //
        size_t                  UsedGPUMemory() const;
        size_t                  UsedCPUMemory() const;

        // Debugging
        void                    GetReadTreeToCPU(DTreeGPU&, std::vector<DTreeNode>&, uint32_t treeIndex) const;
        void                    GetWriteTreeToCPU(DTreeGPU&, std::vector<DTreeNode>&, uint32_t treeIndex) const;
        void                    DumpTreeAsBinary(std::vector<Byte>&, uint32_t& nodeCount, uint32_t treeIndex, bool fetchReadTree) const;

};

inline DTreeGroup::DTreeBuffer::DTreeBuffer()
    : dDTreeNodes(nullptr)
    , dDTreeNodeOffsets(nullptr)
    , dDTrees(nullptr)
{
    hDTreeNodeOffsets.push_back(0);
    offsetMemory = DeviceMemory(sizeof(uint32_t));
    dDTreeNodeOffsets = static_cast<uint32_t*>(offsetMemory);
    CUDA_CHECK(cudaMemset(dDTreeNodeOffsets, 0x00, sizeof(uint32_t)));
}

inline uint32_t DTreeGroup::DTreeBuffer::DTreeCount() const
{
    return static_cast<uint32_t>(hDTreeNodeOffsets.size() - 1);
}

inline uint32_t DTreeGroup::DTreeBuffer::DTreeTotalNodeCount() const
{
    return hDTreeNodeOffsets.back();
}

inline uint32_t DTreeGroup::DTreeBuffer::DTreeNodeCount(uint32_t treeIndex) const
{
    return hDTreeNodeOffsets[treeIndex + 1] - hDTreeNodeOffsets[treeIndex];
}

inline uint32_t DTreeGroup::DTreeBuffer::DTreeNodeOffset(uint32_t treeIndex) const
{
    return hDTreeNodeOffsets[treeIndex];
}

inline const uint32_t* DTreeGroup::DTreeBuffer::DTreeNodeOffsetsGPU() const
{
    return dDTreeNodeOffsets;
}

inline const DTreeGPU* DTreeGroup::DTreeBuffer::DTrees() const
{
    return dDTrees;
}

inline DTreeGPU* DTreeGroup::DTreeBuffer::DTrees()
{
    return dDTrees;
}

inline const DTreeNode* DTreeGroup::DTreeBuffer::DTreeNodesGPU() const
{
    return dDTreeNodes;
}

inline DTreeNode* DTreeGroup::DTreeBuffer::DTreeNodesGPU()
{
    return dDTreeNodes;
}

inline size_t DTreeGroup::DTreeBuffer::UsedGPUMemory() const
{
    return (offsetMemory.Size() +
            treeMemory.Size() +
            treeNodeMemory.Size());
}

inline size_t DTreeGroup::DTreeBuffer::UsedCPUMemory() const
{
    return (sizeof(DTreeBuffer) + hDTreeNodeOffsets.size() * sizeof(uint32_t));
}

inline uint32_t DTreeGroup::TreeCount() const
{
    assert(readTrees.DTreeCount() == writeTrees.DTreeCount());
    return readTrees.DTreeCount();
}

inline size_t DTreeGroup::NodeCount(uint32_t treeIndex, bool readOrWriteTree) const
{
    const DTreeBuffer& treeBuffer = (readOrWriteTree) ? readTrees : writeTrees;
    return treeBuffer.DTreeNodeCount(treeIndex);
}

inline DTreeGPU* DTreeGroup::ReadTrees()
{
    return readTrees.DTrees();
}

inline const DTreeGPU* DTreeGroup::ReadTrees() const
{
    return readTrees.DTrees();
}

inline DTreeGPU* DTreeGroup::WriteTrees()
{
    return writeTrees.DTrees();
}

inline const DTreeGPU* DTreeGroup::WriteTrees() const
{
    return writeTrees.DTrees();
}

inline size_t DTreeGroup::UsedGPUMemory() const
{
    return (writeTrees.UsedGPUMemory() +
            readTrees.UsedGPUMemory());
}
inline size_t DTreeGroup::UsedCPUMemory() const
{
    return (writeTrees.UsedCPUMemory() +
            readTrees.UsedCPUMemory());
}

inline void DTreeGroup::GetReadTreeToCPU(DTreeGPU& tree, std::vector<DTreeNode>& nodes, uint32_t treeIndex) const
{
    readTrees.GetTreeToCPU(tree, nodes, treeIndex);
}

inline void DTreeGroup::GetWriteTreeToCPU(DTreeGPU& tree, std::vector<DTreeNode>& nodes, uint32_t treeIndex) const
{
    writeTrees.GetTreeToCPU(tree, nodes, treeIndex);
}

inline void DTreeGroup::DumpTreeAsBinary(std::vector<Byte>& data, uint32_t& nodeCount,
                                         uint32_t treeIndex, bool fetchReadTree) const
{
    const DTreeBuffer& treeBuffer = (fetchReadTree) ? readTrees : writeTrees;
    treeBuffer.DumpTreeAsBinary(data, nodeCount, treeIndex);
}