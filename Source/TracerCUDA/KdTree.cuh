#pragma once

#include "RayLib/AABB.h"
#include "RayLib/Types.h"
#include "RayLib/Constants.h"
#include "DeviceMemory.h"
#include "CudaSystem.h"

#include <queue>
#include <ostream>

template <class V>
class KDTreeCPU;

template <class V>
class KDTreeGPU
{
    public:
    enum AxisType
    {
        X = 0,
        Y = 1,
        Z = 2,
        AXIS_END = 3
    };

    __host__ __device__
    static uint64_t PackInfo(uint32_t parentIndex,
                             uint32_t childOrLeafIndex,
                             bool isLeaf,
                             AxisType axis);
    __host__ __device__
    static void UnPackInfo(uint32_t& parentIndex,
                           uint32_t& childOrLeafIndex,
                           bool& isLeaf,
                           AxisType& axis,
                           uint64_t data);

    __host__ __device__
    static void UpdateChildIndex(uint64_t& packedData, uint32_t childIndex);

    private:
    static constexpr uint32_t CHILD_BIT_COUNT = 30;
    static constexpr uint32_t PARENT_BIT_COUNT = 30;
    static constexpr uint32_t AXIS_BIT_COUNT = 2;
    static constexpr uint32_t IS_LEAF_BIT_COUNT = 1;

    static constexpr uint64_t CHILD_BIT_MASK = (1ull << CHILD_BIT_COUNT) - 1;
    static constexpr uint64_t PARENT_BIT_MASK = (1ull << PARENT_BIT_COUNT) - 1;
    static constexpr uint64_t AXIS_BIT_MASK = (1ull << AXIS_BIT_COUNT) - 1;
    static constexpr uint64_t IS_LEAF_BIT_MASK = (1ull << IS_LEAF_BIT_COUNT) - 1;

    static constexpr uint32_t CHILD_START = 0;
    static constexpr uint32_t PARENT_START = CHILD_START + CHILD_BIT_COUNT;
    static constexpr uint32_t AXIS_START = PARENT_START + PARENT_BIT_COUNT;
    static constexpr uint32_t IS_LEAF_BIT_START = AXIS_START + AXIS_BIT_COUNT;

    static_assert((CHILD_BIT_COUNT + PARENT_BIT_COUNT +
                   IS_LEAF_BIT_COUNT + AXIS_BIT_COUNT) <= sizeof(uint64_t) * BYTE_BITS,
                  "KDTree Packed Data exceeds 64-bit int size.");

    static constexpr uint8_t MAX_DEPTH = 64;

    friend class KDTreeCPU<V>;

    // Properties
    const float*        gSplits;
    const uint64_t*     gPackedData;
    const V*            gLeafs;
    uint32_t            rootNodeId;
    float               voronoiCenterSize;

    // Helper Functions
    __device__ uint32_t     SelectChild(const V& pos,
                                        uint32_t nodeId) const;
    __device__ bool         IsLeaf(uint32_t nodeId) const;
    //
    __device__ AxisType     Axis(uint32_t nodeId) const;
    __device__ uint32_t     LeftChildId(uint32_t nodeId) const;
    __device__ uint32_t     RightChildId(uint32_t nodeId) const;
    __device__ uint32_t     ParentId(uint32_t nodeId) const;
    __device__ uint32_t     LeafIndex(uint32_t nodeId) const;

    public:
    // Constructors & Destructor
                            KDTreeGPU() = default;
    __device__              KDTreeGPU(const float* gSplits,
                                      const uint64_t* gPackedData,
                                      uint32_t rootNodeId);
                            ~KDTreeGPU() = default;

    __device__ uint32_t     FindNearestPoint(float& distance,
                                             const V& point) const;

    __device__ float        VoronoiCenterSize() const;
};

template <class V>
class KDTreeCPU
{
    private:
        KDTreeGPU<V>        treeGPU;
        DeviceMemory        memory;

        uint32_t            nodeCount;
        uint32_t            leafCount;

        static float        CalculateVoronoiCenterSize(const AABB3f& aabb);

    public:
        // Constructors & Destructor
                            KDTreeCPU();
                            KDTreeCPU(const KDTreeCPU&) = delete;
        KDTreeCPU&          operator=(const KDTreeCPU&) = delete;
                            ~KDTreeCPU() = default;

        // Construct using the leaf list gpu data provided
        TracerError         Construct(const V* dPositionList,
                                      uint32_t leafCount,
                                      const CudaSystem& system);

        const KDTreeGPU<V>& TreeGPU() const;

        size_t              UsedGPUMemory() const;
        size_t              UsedCPUMemory() const;

        void                DumpTreeToStream(std::ostream& stream) const;
        void                DumpTreeAsBinary(std::vector<Byte>& data) const;
};

template <class V>
__host__ __device__ inline
uint64_t KDTreeGPU<V>::PackInfo(uint32_t parentIndex,
                             uint32_t childOrLeafIndex,
                             bool isLeaf,
                             AxisType axis)
{
    uint64_t result = 0;
    result |= (static_cast<uint64_t>(childOrLeafIndex) & CHILD_BIT_MASK) << CHILD_START;
    result |= (static_cast<uint64_t>(parentIndex) & PARENT_BIT_MASK) << PARENT_START;
    result |= (static_cast<uint64_t>(axis) & AXIS_BIT_MASK) << AXIS_START;
    result |= (static_cast<uint64_t>(isLeaf) & IS_LEAF_BIT_MASK) << IS_LEAF_BIT_START;
    return result;
}

template <class V>
__host__ __device__ inline
void KDTreeGPU<V>::UnPackInfo(uint32_t& parentIndex,
                              uint32_t& childOrLeafIndex,
                              bool& isLeaf,
                              AxisType& axis,
                              uint64_t p)
{
    parentIndex = (p >> PARENT_START) & PARENT_BIT_MASK;
    childOrLeafIndex = (p >> CHILD_START) & CHILD_BIT_MASK;
    axis = static_cast<AxisType>((p >> AXIS_START) & AXIS_BIT_MASK);
    isLeaf = (p >> IS_LEAF_BIT_START)& IS_LEAF_BIT_MASK;
}

template <class V>
__host__ __device__ inline
void KDTreeGPU<V>::UpdateChildIndex(uint64_t& packedData, uint32_t childIndex)
{
    uint64_t otherBits = packedData & ~(CHILD_BIT_MASK << CHILD_START);
    uint64_t newBits = (static_cast<uint64_t>(childIndex) & CHILD_BIT_MASK) << CHILD_START;
    packedData = otherBits | newBits;
}

template <class V>
__device__ inline
uint32_t KDTreeGPU<V>::SelectChild(const V& pos, uint32_t nodeId) const
{
    AxisType axis = Axis(nodeId);
    if(gSplits[nodeId] <= pos[static_cast<int>(axis)])
        return LeftChildId(nodeId);
    else
        return RightChildId(nodeId);
}

template <class V>
__device__ inline
bool KDTreeGPU<V>::IsLeaf(uint32_t nodeId) const
{
    uint64_t p = gPackedData[nodeId];
    return (p >> IS_LEAF_BIT_START) & IS_LEAF_BIT_MASK;
}

template <class V>
__device__ inline
KDTreeGPU<V>::AxisType KDTreeGPU<V>::Axis(uint32_t nodeId) const
{
    uint64_t p = gPackedData[nodeId];
    uint32_t axis = (p >> AXIS_START) & AXIS_BIT_MASK;
    return static_cast<AxisType>(axis);
}

template <class V>
__device__ inline
uint32_t KDTreeGPU<V>::LeftChildId(uint32_t nodeId) const
{
    uint64_t p = gPackedData[nodeId];
    return (p >> CHILD_START) & CHILD_BIT_MASK;
}

template <class V>
__device__ inline
uint32_t KDTreeGPU<V>::RightChildId(uint32_t nodeId) const
{
    return LeftChildId(nodeId) + 1;
}

template <class V>
__device__ inline
uint32_t KDTreeGPU<V>::ParentId(uint32_t nodeId) const
{
    uint64_t p = gPackedData[nodeId];
    return (p >> PARENT_START) & PARENT_BIT_MASK;
}

template <class V>
__device__ inline
uint32_t KDTreeGPU<V>::LeafIndex(uint32_t nodeId) const
{
    return LeftChildId(nodeId);
}

template <class V>
__device__ inline
KDTreeGPU<V>::KDTreeGPU(const float* gSplits,
                        const uint64_t* gPackedData,
                        uint32_t rootNodeId)
    : gSplits(gSplits)
    , gPackedData(gPackedData)
    , rootNodeId(rootNodeId)
{}

template <class V>
__device__ inline
uint32_t KDTreeGPU<V>::FindNearestPoint(float& distance,
                                        const V& point) const
{
    struct StackData
    {
        V deltaDist;
        uint32_t index;

    };
    // Check if the stack is properly aligned
    static_assert(sizeof(StackData) == (sizeof(V) + sizeof(uint32_t)),
                  "KdTree traverse stack is not properly aligned!");

    // TODO: Stack is quite thick (1Kib per thread ouch!)
    StackData sLocationStack[MAX_DEPTH];
    // Convenience Functions
    auto Push = [&sLocationStack](uint8_t& depth, uint32_t loc,
                                  const V& deltaDist) -> void
    {
        assert(depth < MAX_DEPTH);
        uint32_t index = depth;
        sLocationStack[index].index = loc;
        sLocationStack[index].deltaDist = deltaDist;
        depth++;
    };
    auto ReadTop = [&sLocationStack](uint8_t depth) -> StackData
    {
        assert(depth > 0);
        return sLocationStack[depth - 1];
    };
    auto Pop = [&sLocationStack](uint8_t& depth) -> StackData
    {
        assert(depth > 0);
        depth--;
        return sLocationStack[depth];
    };

    // Initialize
    distance = FLT_MAX;
    uint32_t resultNodeIndex = UINT32_MAX;

    // First Descent down on the tree
    uint32_t currentNode = rootNodeId;
    while(!IsLeaf(currentNode))
    {
        // Descent down
        uint32_t childId = SelectChild(point, currentNode);
        currentNode = childId;
    }
    // Calculate the distance
    uint32_t leafIndex = LeafIndex(currentNode);
    V leafPos = gLeafs[leafIndex];
    distance = (point - leafPos).LengthSqr();
    resultNodeIndex = currentNode;

    // We found a leaf, which maybe the closest
    // But we need to re-traverse the tree
    // to find the actual closest distance
    //
    // This implementation is quite similar to the
    // https://github.com/NVlabs/fermat/blob/master/contrib/cugar/kd/cuda/knn_inline.h

    // Re-traverse the tree using stack
    uint8_t depth = 0;
    Push(depth, rootNodeId, V(0.0f));
    while(depth > 0)
    {
        auto traverseData = Pop(depth);
        uint32_t node = traverseData.index;

        if(IsLeaf(node))
        {
            // Calculate the distance
            uint32_t leafIndex = LeafIndex(node);
            V leafPos = gLeafs[leafIndex];
            float dist = (point - leafPos).LengthSqr();
            // Update close point
            if(dist < distance)
            {
                distance = dist;
                resultNodeIndex = node;
            }
            // Pop unnecessary nodes in the stack
            // After the update, some nodes are not necessary
            // Pop those until stack is either empty or we found
            // valid stack entry
            while(depth > 0 ||
                  ReadTop(depth).deltaDist.LengthSqr() >= distance)
            {
                Pop(depth);
            }
        }
        else
        {
            // Fetch Current Split Plane
            int splitAxis = static_cast<int>(Axis(node));
            float splitPlane = gSplits[node];
            float splitDist = point[splitAxis] - splitPlane;

            // Generate Far point
            V deltaDist = traverseData.deltaDist;
            deltaDist[splitAxis] = splitDist;

            // Select Child
            uint32_t childId = SelectChild(point, node);
            uint32_t otherChildId = (childId == LeftChildId(node))
                                        ? RightChildId(node)
                                        : LeftChildId(node);

            Push(depth, childId, traverseData.deltaDist);
            // Push the non-selected child as well
            // If it may contain a closer point
            if(deltaDist.LengthSqr() < distance)
                Push(depth, otherChildId, deltaDist);
        }
    }

    // Calculate actual distance
    distance = sqrt(distance);
    return LeafIndex(resultNodeIndex);
}

template <class V>
__device__ inline
float KDTreeGPU<V>::VoronoiCenterSize() const
{
    return voronoiCenterSize;
}

#include "KDTree.hpp"

extern template class KDTreeCPU<Vector3f>;
extern template class KDTreeGPU<Vector3f>;