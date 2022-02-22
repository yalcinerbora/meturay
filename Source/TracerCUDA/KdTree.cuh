#pragma once

#include "RayLib/AABB.h"
#include "RayLib/Types.h"
#include "DeviceMemory.h"
#include "CudaSystem.h"

class KDTreeCPU;

class KDTreeGPU
{
    public:
    enum AxisType
    {
        X,
        Y,
        Z,
        AXIS_END
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

    static constexpr uint32_t CHILD_START = 0;
    static constexpr uint32_t PARENT_START = CHILD_START + CHILD_BIT_COUNT;
    static constexpr uint32_t AXIS_START = PARENT_START + PARENT_BIT_COUNT;
    static constexpr uint32_t IS_LEAF_BIT_START = AXIS_START + AXIS_BIT_COUNT;

    static_assert((CHILD_BIT_COUNT + PARENT_BIT_COUNT +
                   IS_LEAF_BIT_COUNT + AXIS_BIT_COUNT) <= sizeof(uint64_t) * BYTE_BITS,
                  "KDTree Packed Data exceeds 64-bit int size.");

    static constexpr uint8_t MAX_DEPTH = 64;

    friend class KDTreeCPU;

    // Properties
    const float*        gSplits;
    const uint64_t*     gPackedData;
    const Vector3f*     gLeafs;
    uint32_t            rootNodeId;
    float               voronoiCenterSize;

    // Helper Functions
    __device__ uint32_t     SelectChild(const Vector3f& pos,
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
                                             const Vector3f& point) const;

    __device__ float        VoronoiCenterSize() const;
};

class KDTreeCPU
{
    private:
        KDTreeGPU           treeGPU;
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
        TracerError         Construct(const Vector3f* dPositionList,
                                      uint32_t leafCount,
                                      const CudaSystem& system);

        const KDTreeGPU&    TreeGPU() const;

        size_t              UsedGPUMemory() const;
        size_t              UsedCPUMemory() const;

        void                DumpTreeToStream(std::ostream& stream) const;
        void                DumpTreeAsBinary(std::vector<Byte>& data) const;
};

__host__ __device__ inline
uint64_t KDTreeGPU::PackInfo(uint32_t parentIndex,
                             uint32_t childOrLeafIndex,
                             bool isLeaf,
                             AxisType axis)
{
    assert(parentIndex < (1 << PARENT_BIT_COUNT));
    assert(childOrLeafIndex < (1 << CHILD_BIT_COUNT));

    uint64_t result = 0;
    result |= (static_cast<uint64_t>(childOrLeafIndex) << CHILD_START);
    result |= (static_cast<uint64_t>(parentIndex) << PARENT_START);
    result |= (static_cast<uint64_t>(axis) << AXIS_START);
    result |= (static_cast<uint64_t>(isLeaf) << IS_LEAF_BIT_START);
    return result;
}

__host__ __device__ inline
void KDTreeGPU::UnPackInfo(uint32_t& parentIndex,
                           uint32_t& childOrLeafIndex,
                           bool& isLeaf,
                           AxisType& axis,
                           uint64_t p)
{
    parentIndex = (p >> PARENT_START) & PARENT_BIT_COUNT;
    childOrLeafIndex = (p >> CHILD_START) & CHILD_BIT_COUNT;
    axis = static_cast<AxisType>((p >> AXIS_START) & AXIS_BIT_COUNT);
    isLeaf = (p >> IS_LEAF_BIT_START)& IS_LEAF_BIT_COUNT;
}

__host__ __device__ inline
void KDTreeGPU::UpdateChildIndex(uint64_t& packedData, uint32_t childIndex)
{
    packedData |= (static_cast<uint64_t>(childIndex) << CHILD_START);
}

__device__ inline
uint32_t KDTreeGPU::SelectChild(const Vector3f& pos,
                                uint32_t nodeId) const
{
    AxisType axis = Axis(nodeId);
    if(gSplits[nodeId] <= pos[static_cast<int>(axis)])
        return LeftChildId(nodeId);
    else
        return RightChildId(nodeId);
}

__device__ inline
bool KDTreeGPU::IsLeaf(uint32_t nodeId) const
{
    uint64_t p = gPackedData[nodeId];
    return (p >> IS_LEAF_BIT_START) & IS_LEAF_BIT_COUNT;
}

__device__ inline
KDTreeGPU::AxisType KDTreeGPU::Axis(uint32_t nodeId) const
{
    uint64_t p = gPackedData[nodeId];
    uint32_t axis = (p >> AXIS_START) & AXIS_BIT_COUNT;
    return static_cast<AxisType>(axis);
}

__device__ inline
uint32_t KDTreeGPU::LeftChildId(uint32_t nodeId) const
{
    uint64_t p = gPackedData[nodeId];
    return (p >> CHILD_START) & CHILD_BIT_COUNT;
}

__device__ inline
uint32_t KDTreeGPU::RightChildId(uint32_t nodeId) const
{
    return LeftChildId(nodeId) + 1;
}

__device__ inline
uint32_t KDTreeGPU::ParentId(uint32_t nodeId) const
{
    uint64_t p = gPackedData[nodeId];
    return (p >> PARENT_START) & PARENT_BIT_COUNT;
}

__device__ inline
uint32_t KDTreeGPU::LeafIndex(uint32_t nodeId) const
{
    return LeftChildId(nodeId);
}

__device__ inline
KDTreeGPU::KDTreeGPU(const float* gSplits,
                     const uint64_t* gPackedData,
                     uint32_t rootNodeId)
    : gSplits(gSplits)
    , gPackedData(gPackedData)
    , rootNodeId(rootNodeId)
{}

__device__ inline
uint32_t KDTreeGPU::FindNearestPoint(float& distance,
                                     const Vector3f& point) const
{
    struct StackData
    {
        Vector3f position;
        uint32_t index;

    };
    StackData sLocationStack[MAX_DEPTH];

    // Convenience Functions
    auto Push = [&sLocationStack](uint8_t& depth, uint32_t loc,
                                  const Vector3f& position) -> void
    {
        uint32_t index = depth;
        sLocationStack[index].index = loc;
        sLocationStack[index].position = position;
        depth++;
    };
    auto ReadTop = [&sLocationStack](uint8_t depth) -> StackData
    {
        uint32_t index = depth;
        return sLocationStack[index];
    };
    auto Pop = [&ReadTop](uint8_t& depth) -> StackData
    {
        depth--;
        return ReadTop(depth);
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
    Vector3f leafPos = gLeafs[LeafIndex(currentNode)];
    distance = (point - leafPos).LengthSqr();
    resultNodeIndex = currentNode;

    return resultNodeIndex;

    // We found a leaf, which maybe the closest
    // But we need to back-propagate towards to the root
    // (re-descent if necessary) and find the actual closest distance
    // Or we can descent down again
    // And check other points using a stack
    //
    // This implementation is quite similar to the
    // https://github.com/NVlabs/fermat/blob/master/contrib/cugar/kd/cuda/knn_inline.h


    // We already on a leaf with the appropriate stack info
    // Continue traversing from the parent
    uint8_t depth = 0;
    Push(depth, rootNodeId, Vector3f(0.0f));

    currentNode = rootNodeId;
    while(depth > 0)
    {
        auto traverseData = Pop(depth);

        if(IsLeaf(currentNode))
        {
            // Calculate the distance
            uint32_t leafIndex = LeafIndex(currentNode);
            Vector3f leafPos = gLeafs[LeafIndex(currentNode)];
            float dist = (point - leafPos).LengthSqr();

            if(dist < distance)
            {
                distance = dist;
                resultNodeIndex = currentNode;
            }
            // Pop unnecessary nodes in the stack
            // After the update
            while(depth > 0)
            {
                float lSqr = ReadTop(depth).position.LengthSqr();
                // This is necessary break
                if(lSqr < distance) break;

                // Unnecessary data, break
                Pop(depth);
            }
        }
        else
        {
            // Fetch Current Split Plane
            int axis = static_cast<int>(Axis(currentNode));
            float splitPlane = gSplits[axis];
            float splitDist = point[axis] - splitPlane;

            // Generate Far point
            Vector3f farPoint = traverseData.position;
            farPoint[axis] = splitDist;

            // Select Child
            uint32_t childId = SelectChild(point, currentNode);

            // Push the non-selected child as well
            // If it may contain a closer point
            if(farPoint.LengthSqr() < distance)
            {
                Push(depth,
                     (childId == LeftChildId(currentNode))
                        ? RightChildId(currentNode)
                        : LeftChildId(currentNode),
                     farPoint);
            }

            // Descent down
            currentNode = childId;
        }
    }
    return resultNodeIndex;
}

__device__ inline
float KDTreeGPU::VoronoiCenterSize() const
{
    return voronoiCenterSize;
}