#pragma once

#include <cstdint>
#include <queue>

#include "RayLib/AABB.h"

#include "DeviceMemory.h"
#include "CudaSystem.h"
#include "CudaSystem.hpp"
// Algorithms and functors required for the operations
#include "MortonCode.cuh"
#include "ParallelReduction.cuh"
#include "ParallelSequence.cuh"
#include "ParallelRadixSort.cuh"
#include "LinearBVHKC.cuh"
#include "GPUAcceleratorCommonKC.cuh"

#include "TracerDebug.h"

template <class Leaf>
using AABBGenFunc = AABB3f(&)(const Leaf&);

template<class Leaf>
struct /*alignas(16)*/ LBVHNode
{
    // Pointers
    union
    {
        // Non-leaf part
        struct
        {
            Vector3     aabbMin;
            uint32_t    left;
            Vector3     aabbMax;
            uint32_t    right;
        } body;
        // leaf part
        Leaf leaf;
    };
    uint32_t    parent;
    bool        isLeaf;
};

// SFINAE
template <typename T, typename = Vector3f>
struct HasPosition : std::false_type {};

template <typename T>
struct HasPosition<T, decltype(T::position)> : std::true_type {};

template <class Leaf, class DistFunctor>
struct LinearBVHGPU
{
    private:
    __device__ float        InitClosestDistance(const Leaf& worldSurface) const;

    public:
    static constexpr const  uint8_t MAX_DEPTH = 64;

    DistFunctor             DistanceFunction;
    uint32_t                rootIndex;
    const LBVHNode<Leaf>*   nodes;
    uint32_t                nodeCount;
    uint32_t                leafCount;

    __device__ uint32_t     FindNearestPointWithStack(float& distance,
                                                      const Leaf& worldSurface) const;
    __device__ uint32_t     FindNearestPoint(float& distance,
                                             const Leaf& worldSurface) const;
    __device__ float        VoronoiCenterSize() const;
};

template <class Leaf,
          class DistFunctor,
          AABBGenFunc<Leaf> AF>
class LinearBVHCPU
{
    public:
        using DeviceBVH                 = LinearBVHGPU<Leaf, DistFunctor>;
        static constexpr auto AABBFunc  = AF;

    private:
        DeviceBVH           treeGPU;
        DeviceMemory        memory;

        static float        FindMortonDelta(const AABB3f& extent);

    protected:
    public:
        // Constructors & Destructor
                            LinearBVHCPU();
                            LinearBVHCPU(const LinearBVHCPU&) = delete;
        LinearBVHCPU&       operator=(const LinearBVHCPU&) = delete;
                            ~LinearBVHCPU() = default;

        // Construct using the leaf list gpu data provided
        TracerError         Construct(const Leaf* dLeafList,
                                      uint32_t leafCount,
                                      DistFunctor df,
                                      const CudaSystem& system);
        TracerError         ConstructNonLinear(const Leaf* dLeafList,
                                               uint32_t leafCount,
                                               DistFunctor df,
                                               const CudaSystem& system);

        // Getters
        uint32_t            NodeCount() const;
        uint32_t            LeafCount() const;
        const DeviceBVH&    TreeGPU() const;

        size_t              UsedGPUMemory() const;
        size_t              UsedCPUMemory() const;

        void                DumpTreeAsBinary(std::vector<Byte>& data) const;
};

template <class Leaf, class DistFunctor>
__device__ inline
float LinearBVHGPU<Leaf, DistFunctor>::InitClosestDistance(const Leaf& worldSurface) const
{
    auto DetermineDistance = [&](const LBVHNode<Leaf>* childNode)
    {
        float childDistance = FLT_MAX;
        if(childNode->isLeaf)
        {
            childDistance = DistanceFunction(childNode->leaf, worldSurface);
        }
        else
        {
            AABB3f aabbChild = AABB3f(childNode->body.aabbMin,
                                      childNode->body.aabbMax);
            if(aabbChild.IsInside(worldSurface.position))
            {
                childDistance = 0.0f;
            }
            else
            {
                childDistance = aabbChild.FurthestCorner(worldSurface.position).Length()
                    + MathConstants::Epsilon;
            }
        }
        return childDistance;
    };
    // Utilize BVH as a Kd Tree and do AABB-Point
    // intersection instead of AABB-Sphere.
    // This is not an exact solution and may fail
    const LBVHNode<Leaf>* currentNode = nodes + rootIndex;
    // Descent towards the tree
    while(!currentNode->isLeaf)
    {
        // Check both child here
        const LBVHNode<Leaf>* leftNode = nodes + currentNode->body.left;
        const LBVHNode<Leaf>* rightNode = nodes + currentNode->body.right;
        // Determine the distances
        float leftDistance = DetermineDistance(leftNode);
        float rightDistance = DetermineDistance(rightNode);
        // Select the closest child
        currentNode = (leftDistance < rightDistance) ? leftNode : rightNode;
    }
    // We found a leaf so use it
    return DistanceFunction(currentNode->leaf, worldSurface) + MathConstants::Epsilon;

}

template <class Leaf, class DistFunctor>
__device__ inline
uint32_t LinearBVHGPU<Leaf, DistFunctor>::FindNearestPointWithStack(float& distance, const Leaf& worldSurface) const
{
    static_assert(HasPosition<Leaf>::value,
                  "This functions requires its leafs to have public \"position\" variable");

    // Minimal stack to traverse
    uint32_t sLocationStack[MAX_DEPTH];
    // Convenience Functions
    auto Push = [&sLocationStack](uint8_t& depth, uint32_t loc) -> void
    {
        uint32_t index = depth;
        sLocationStack[index] = loc;
        depth++;
    };
    auto ReadTop = [&sLocationStack](uint8_t depth) -> uint32_t
    {
        uint32_t index = depth;
        return sLocationStack[index];
    };
    auto Pop = [&ReadTop](uint8_t& depth) -> uint32_t
    {
        depth--;
        return ReadTop(depth);
    };

    // Initialize with an approximate closest value
    float closestDistance = InitClosestDistance(worldSurface);
    uint32_t closestLeafIndex = UINT32_MAX;
    // TODO: There is an optimization here
    // first iteration until leaf is always true
    // initialize closest distance with the radius
    uint8_t depth = 0;
    Push(depth, rootIndex);
    const LBVHNode<Leaf>* currentNode = nullptr;
    while(depth > 0)
    {
        uint32_t loc = Pop(depth);
        currentNode = nodes + loc;

        if(currentNode->isLeaf)
        {
            float distance = DistanceFunction(currentNode->leaf, worldSurface);
            if(distance < closestDistance)
            {
                closestDistance = distance;
                closestLeafIndex = currentNode->leaf.leafId;
            }
        }
        else if(AABB3f aabb = AABB3f(currentNode->body.aabbMin,
                                     currentNode->body.aabbMax);
                aabb.IntersectsSphere(worldSurface.position,
                                      closestDistance))
        {
            // Push to stack
            Push(depth, currentNode->body.right);
            Push(depth, currentNode->body.left);
        }
    }
    distance = closestDistance;
    return closestLeafIndex;
}

template <class Leaf, class DistFunctor>
__device__ inline
uint32_t LinearBVHGPU<Leaf, DistFunctor>::FindNearestPoint(float& distance, const Leaf& worldSurface) const
{
    static_assert(HasPosition<Leaf>::value,
                  "This functions requires its leafs to have public \"position\" variable");

    // Helper Variables
    static constexpr uint8_t FIRST_ENTRY = 0b00;
    static constexpr uint8_t U_TURN = 0b01;
    // Convenience Functions
    auto WipeLowerBits = [](uint64_t& list, uint8_t depth)-> void
    {
        depth--;
        list &= (UINT64_MAX << depth);
    };
    auto TraverseInfo = [](uint64_t list, uint8_t depth) -> uint8_t
    {
        depth -= 2;
        return static_cast<uint8_t>((list >> depth) & 0x3ull);
    };
    auto MarkAsTraversed = [](uint64_t& list, uint8_t depth) -> void
    {
        depth--;
        list += (1ull << depth);
    };
    auto Pop = [&MarkAsTraversed](uint64_t& list, uint8_t& depth) -> void
    {
        MarkAsTraversed(list, depth);
        depth++;
    };
    // Resulting Closest Leaf Index
    // & Closest Hit
    // Initialize with an approximate closest value
    float closestDistance = InitClosestDistance(worldSurface);
    uint32_t closestLeafIndex = UINT32_MAX;

    // Bit Stack and its pointer
    uint64_t list = 0;
    uint8_t depth = MAX_DEPTH;
    // Root Node
    const LBVHNode<Leaf>* currentNode = nodes + rootIndex;
    while(depth <= MAX_DEPTH)
    {
        // Determine traverse information
        uint8_t info = TraverseInfo(list, depth);
        // First time entry check intersection
        if(info == FIRST_ENTRY)
        {
            // If Leaf, do its custom closest function primitive intersection
            if(currentNode->isLeaf)
            {
                //printf("Leaf, CheckingDistance\n");
                float distance = DistanceFunction(currentNode->leaf, worldSurface);
                if(distance < closestDistance)
                {
                    closestDistance = distance;
                    closestLeafIndex = currentNode->leaf.leafId;
                }
                // Go Up
                currentNode = nodes + currentNode->parent;
                Pop(list, depth);
            }
            // If not leaf check if there is closer point in this AABB
            // meaning that a sphere defined by "worldPos, closestDistance"
            // intersects with this AABB
            else if(AABB3f aabb = AABB3f(currentNode->body.aabbMin,
                                         currentNode->body.aabbMax);
                    aabb.IntersectsSphere(worldSurface.position,
                                          closestDistance))
            {
                // By construction BVH tree has either no or both children
                // avail. If a node is non-leaf it means that it has both of its children
                // no need to check for left or right index validity

                // Directly go left
                currentNode = nodes + currentNode->body.left;
                depth--;
            }
            // If we could not be able to hit AABB
            // just go parent
            else
            {
                //printf("AABB Not Hit\n");
                currentNode = nodes + currentNode->parent;
                Pop(list, depth);
            }
        }
        // Doing U turn (left to right)
        // Here we came from left and need to go to right
        // we are at parent
        else if(info == U_TURN)
        {
            // Go to right child if avail
            MarkAsTraversed(list, depth - 1);
            currentNode = nodes + currentNode->body.right;
            depth--;
        }
        // Now both left and right are processed
        // Just go up
        else
        {
            // Wipe out lower bits for incoming iterations
            WipeLowerBits(list, depth);
            currentNode = nodes + currentNode->parent;
            depth++;
        }
    }
    distance = closestDistance;
    return closestLeafIndex;
}

template <class Leaf, class DistFunctor>
__device__ inline
float LinearBVHGPU<Leaf, DistFunctor>::VoronoiCenterSize() const
{
    const AABB3f sceneAABB(nodes[rootIndex].body.aabbMin,
                           nodes[rootIndex].body.aabbMax);
    Vector3f span = sceneAABB.Span();
    float sceneSize = span.Length();
    static constexpr float VORONOI_RATIO = 1.0f / 1'300.0f;
    return sceneSize * VORONOI_RATIO;
}

template <class Leaf, class DF,
          AABBGenFunc<Leaf> AF>
inline uint32_t LinearBVHCPU<Leaf, DF, AF>::NodeCount() const
{
    return treeGPU.nodeCount;
}

template <class Leaf, class DF,
          AABBGenFunc<Leaf> AF>
inline uint32_t LinearBVHCPU<Leaf, DF, AF>::LeafCount() const
{
    return treeGPU.leafCount;
}

template <class Leaf, class DF,
          AABBGenFunc<Leaf> AF>
inline const LinearBVHGPU<Leaf, DF>& LinearBVHCPU<Leaf, DF, AF>::TreeGPU() const
{
    return treeGPU;
}

template <class Leaf, class DF,
          AABBGenFunc<Leaf> AF>
inline size_t LinearBVHCPU<Leaf, DF, AF>::UsedGPUMemory() const
{
    return memory.Size();
}

template <class Leaf, class DF,
          AABBGenFunc<Leaf> AF>
inline size_t LinearBVHCPU<Leaf, DF, AF>::UsedCPUMemory() const
{
    return sizeof(LinearBVHCPU);
}

// Basic Definitions for point query in Linear BVH
extern template struct LBVHNode<PointStruct>;
extern template struct LinearBVHGPU<PointStruct, PointDistanceFunctor>;
extern template class LinearBVHCPU<PointStruct, PointDistanceFunctor, GenPointAABB>;

using LBVHPointGPU = LinearBVHGPU<PointStruct, PointDistanceFunctor>;
using LBVHPointCPU = LinearBVHCPU<PointStruct, PointDistanceFunctor, GenPointAABB>;

template <class T>
std::ostream& operator<<(std::ostream& s, const LBVHNode<T>& n);

#include "LinearBVH.hpp"