#pragma once

#include <cstdint>

#include "RayLib/Ray.h"
#include "RayLib/AABB.h"
#include "GPULightI.h"
#include "RayLib/TracerStructs.h"
#include "DeviceMemory.h"

struct PathGuidingNode;
class CudaSystem;

class AnisoSVOctreeGPU
{
    private:
    template <class T>
    struct AnisoData
    {
        Vector<4, T> data[2];
    };
    using AnisoRadiance = AnisoData<half>;
    using AnisoRadianceF = AnisoData<float>;
    using AnisoCount = AnisoData<uint32_t>;

    static constexpr uint64_t IS_LEAF_BIT_COUNT     = 1;
    static constexpr uint64_t CHILD_MASK_BIT_COUNT  = 8;
    static constexpr uint64_t PARENT_BIT_COUNT      = 28;
    static constexpr uint64_t CHILD_BIT_COUNT       = 27;

    static constexpr uint64_t CHILD_OFFSET          = 0;
    static constexpr uint64_t PARENT_OFFSET         = CHILD_OFFSET + CHILD_BIT_COUNT;
    static constexpr uint64_t CHILD_MASK_OFFSET     = PARENT_OFFSET + PARENT_BIT_COUNT;
    static constexpr uint64_t IS_LEAF_OFFSET        = CHILD_MASK_OFFSET + CHILD_MASK_BIT_COUNT;

    static constexpr uint64_t IS_LEAF_BIT_MASK      = (1 << IS_LEAF_BIT_COUNT) - 1;
    static constexpr uint64_t PARENT_BIT_MASK       = (1 << PARENT_BIT_COUNT) - 1;
    static constexpr uint64_t CHILD_BIT_MASK        = (1 << CHILD_BIT_COUNT) - 1;
    static constexpr uint64_t CHILD_MASK_BIT_MASK   = (1 << CHILD_MASK_BIT_COUNT) - 1;
    // Sanity Check
    static_assert(sizeof(uint64_t)* BYTE_BITS == (IS_LEAF_BIT_COUNT +
                                                  PARENT_BIT_COUNT +
                                                  CHILD_BIT_COUNT +
                                                  CHILD_MASK_BIT_COUNT),
                  "SVO Packed Bits exceeds 64-bit uint");
    // Data Unpack
    __device__ static bool     IsChildrenLeaf(uint64_t packedData);
    __device__ static uint8_t  ChildMask(uint64_t packedData);
    __device__ static uint32_t ChildrenIndex(uint64_t packedData);
    __device__ static uint32_t ParentIndex(uint64_t packedData);
    // Data Pack
    __device__ static void     SetIsChildrenLeaf(uint64_t& packedData, bool);
    __device__ static void     SetChildMask(uint64_t& packedData, uint8_t);
    __device__ static void     SetChildrenIndex(uint64_t& packedData, uint32_t);
    __device__ static void     SetParentIndex(uint64_t& packedData, uint32_t);

    // SVO Data
    uint64_t*           dNodes;     // children ptr (28) parent ptr (27), child mask, leafBit
    AnisoRadiance*      dRadiance;  // Anisotropic emitted radiance (normalized)
    uint32_t*           dRayCounts; // MSB is "isCollapsed" bit
    // Leaf Data (Leafs does not have nodes,
    // if leaf bit is set on parent children ptr points to these)
    uint32_t*           dLeafParents;
    AnisoRadiance*      dLeafRadianceRead;  // Anisotropic emitted radiance (normalized)
    uint32_t*           dLeafRayCounts;     // MSB is "isCollapsed" bit

    // Actual average radiance for each location (used full precision here for running average)
    AnisoRadianceF*     dLeafRadianceWrite;
    AnisoCount*         dLeafSampleCountWrite;
    // Boundary Light (Used when any ray does not hit anything)
    const GPULightI*    dBoundaryLight;
    // Constants
    AABB3f              sceneAABB;
    uint32_t            voxelResolution;    // x = y = z
    uint32_t            leafDepth;          // log2(voxelResolution)
    uint32_t            nodeCount;
    uint32_t            leafCount;
    float               leafVoxelSize;
    // CPU class can only access and set the data
    friend class        AnisoSVOctreeCPU;

    public:
    // Constructors & Destructor
    __device__          AnisoSVOctreeGPU();
    // Methods
    // Trace the ray over the SVO and find the radiance towards the rays origin
    __device__
    float               TraceRay(const RayF&) const;
    // Deposit radiance to the nearest voxel leaf
    // Use atomics, returns false if no leaf is found on this location
    __device__
    bool                DepositRadiance(const Vector3f& worldPos, const Vector3f& outgoingDir,
                                        float radiance);
    // Return the leaf index of this position
    // Return false if no such leaf exists
    __device__
    bool                LeafIndex(uint32_t& index, const Vector3f& worldPos) const;
    // Find the bin from the leaf
    // Bin is the node that is the highest non-collapsed node
    __device__
    uint32_t            FindBin(bool& isLeaf, uint32_t upperLimit, uint32_t leafIndex)  const;

    // Accessors
    __device__
    uint32_t            LeafVoxelSize() const;
    __device__
    uint32_t            LeafCount() const;
    __device__
    uint32_t            NodeCount() const;
    __device__
    float               VoxelResolution() const;
    __device__
    AABB3f              SceneAABB() const;
};

class AnisoSVOctreeCPU
{
    private:
    DeviceMemory                octreeMem;
    AnisoSVOctreeGPU            treeGPU;
    // Level Ranges
    std::vector<Vector2ui>      levelRanges; // Node Range of each level (Except leaf)

    public:
    // Construct SVO using the primitives on the Accelerators
    // initialize the radiances with the Lights
    void                    Constrcut(const AABB3f& sceneAABB, uint32_t resolutionXYZ,
                                      const AcceleratorBatchMap&,
                                      const GPULightI** dSceneLights,
                                      const CudaSystem&);
    // Normalize and filter radiances for sampling
    void                    NormalizeAndFilterRadiance(const CudaSystem&);
    // Collapse the ray counts to find optimal binning
    void                    CollapseRayCounts(uint32_t minLevel,
                                              uint32_t minRayCount,
                                              const CudaSystem&);
    // Accumulate the Emissive Radiance from the paths
    void                    AccumulateRaidances(const PathGuidingNode* dPGNodes,
                                                uint32_t totalNodeCount,
                                                uint32_t maxPathNodePerRay,
                                                const CudaSystem&);
    // Clear the ray counts for the next iteration
    void                    ClearRayCounts(const CudaSystem&);
    // GPU Data Struct Access
    AnisoSVOctreeGPU        TreeGPU();
    // Misc.
    size_t                  UsedGPUMemory() const;
    size_t                  UsedCPUMemory() const;
};

__device__ inline
AnisoSVOctreeGPU::AnisoSVOctreeGPU()
    : dNodes(nullptr)
    , dRadiance(nullptr)
    , dRayCounts(nullptr)
    , dLeafParents(nullptr)
    , dLeafRadianceRead(nullptr)
    , dLeafRayCounts(nullptr)
    , dLeafRadianceWrite(nullptr)
    , dLeafSampleCountWrite(nullptr)
    , dBoundaryLight(nullptr)
    , sceneAABB(Vector3f(0.0f), Vector3f(0.0f))
    , voxelResolution(0)
    , nodeCount(0)
    , leafCount(0)
    , leafVoxelSize(0.0f)
{}

__device__ inline
bool AnisoSVOctreeGPU::IsChildrenLeaf(uint64_t packedData)
{
    return static_cast<bool>((packedData >> IS_LEAF_OFFSET) & IS_LEAF_BIT_COUNT);
}

__device__ inline
uint8_t AnisoSVOctreeGPU::ChildMask(uint64_t packedData)
{
    return static_cast<uint8_t>((packedData >> CHILD_MASK_OFFSET) & CHILD_MASK_BIT_MASK);
}

__device__ inline
uint32_t AnisoSVOctreeGPU::ChildrenIndex(uint64_t packedData)
{
    return static_cast<uint32_t>((packedData >> CHILD_OFFSET) & CHILD_BIT_MASK);
}

__device__ inline
uint32_t AnisoSVOctreeGPU::ParentIndex(uint64_t packedData)
{
    return static_cast<uint32_t>((packedData >> PARENT_OFFSET) & PARENT_BIT_MASK);
}

__device__ inline
void AnisoSVOctreeGPU::SetIsChildrenLeaf(uint64_t& packedData, bool b)
{
    static constexpr uint64_t NEGATIVE_MASK = ~(IS_LEAF_BIT_MASK << IS_LEAF_OFFSET);
    packedData &= NEGATIVE_MASK;
    packedData |= (static_cast<uint64_t>(b) << IS_LEAF_OFFSET);
}

__device__ inline
void AnisoSVOctreeGPU::SetChildMask(uint64_t& packedData, uint8_t mask)
{
    static constexpr uint64_t NEGATIVE_MASK = ~(CHILD_MASK_BIT_MASK << CHILD_MASK_OFFSET);
    packedData &= NEGATIVE_MASK;
    packedData |= (static_cast<uint64_t>(mask) << CHILD_MASK_OFFSET);
}

__device__ inline
void AnisoSVOctreeGPU::SetChildrenIndex(uint64_t& packedData, uint32_t childIndex)
{
    static constexpr uint64_t NEGATIVE_MASK = ~(CHILD_BIT_MASK << CHILD_OFFSET);
    packedData &= NEGATIVE_MASK;
    packedData |= (static_cast<uint64_t>(childIndex) << CHILD_OFFSET);
}

__device__ inline
void AnisoSVOctreeGPU::SetParentIndex(uint64_t& packedData, uint32_t parentIndex)
{
    static constexpr uint64_t NEGATIVE_MASK = ~(PARENT_BIT_MASK << PARENT_OFFSET);
    packedData &= NEGATIVE_MASK;
    packedData |= (static_cast<uint64_t>(parentIndex) << PARENT_OFFSET);
}

__device__ inline
float AnisoSVOctreeGPU::TraceRay(const RayF& ray) const
{
    return 0.0f;
}

__device__ inline
bool AnisoSVOctreeGPU::DepositRadiance(const Vector3f& worldPos,
                                       const Vector3f& outgoingDir,
                                       float radiance)
{
    uint32_t lIndex;
    bool leafFound = LeafIndex(lIndex, worldPos);

    // Extrapolate the data to the all appropriate locations

    return false;
}

__device__ inline
bool AnisoSVOctreeGPU::LeafIndex(uint32_t& index, const Vector3f& worldPos) const
{
    if(sceneAABB.IsOutside(worldPos)) return false;
    // Calculate Dense Voxel Id
    Vector3f lIndex = ((worldPos - sceneAABB.Min()) / leafVoxelSize).Round();
    Vector3ui denseIndex = Vector3ui(lIndex[0], lIndex[1], lIndex[2]);

    // Now descend down
    uint64_t currentNode = dNodes[0];
    for(uint32_t i = 1; i <= leafDepth; i++)
    {
        uint8_t childMask = ChildMask(currentNode);
        uint32_t childPtr = ChildrenIndex(currentNode);

        uint32_t x = (denseIndex[0] >> (leafDepth - i)) & 0b1;
        uint32_t y = (denseIndex[0] >> (leafDepth - i)) & 0b1;
        uint32_t z = (denseIndex[0] >> (leafDepth - i)) & 0b1;
        uint32_t nextChildOffset = 0;
        nextChildOffset |= (z << 2);
        nextChildOffset |= (y << 1);
        nextChildOffset |= (x << 0);

        // Check if this node has that child avail
        if(!(childMask >> nextChildOffset))
        {
            index = UINT32_MAX;
            return false;
        }
        // Continue
        currentNode = dNodes[childPtr + nextChildOffset];
    }
    // If we descended down properly currentNode should point to the
    // index. Notice that, last level's children ptr will point to the leaf arrays
    index = currentNode;
    return true;
}

__device__ inline
uint32_t AnisoSVOctreeGPU::FindBin(bool& isLeaf, uint32_t upperLimit, uint32_t leafIndex) const
{
    return UINT32_MAX;
}

__device__ inline
uint32_t AnisoSVOctreeGPU::LeafVoxelSize() const
{
    return leafVoxelSize;
}

__device__ inline
uint32_t AnisoSVOctreeGPU::LeafCount() const
{
    return leafCount;
}

__device__ inline
uint32_t AnisoSVOctreeGPU::NodeCount() const
{
    return nodeCount;
}

__device__ inline
float AnisoSVOctreeGPU::VoxelResolution() const
{
    return voxelResolution;
}

__device__ inline
AABB3f AnisoSVOctreeGPU::SceneAABB() const
{
    return sceneAABB;
}

inline
AnisoSVOctreeGPU AnisoSVOctreeCPU::TreeGPU()
{
    return treeGPU;
}

inline
size_t AnisoSVOctreeCPU::UsedGPUMemory() const
{
    return octreeMem.Size();
}

inline
size_t AnisoSVOctreeCPU::UsedCPUMemory() const
{
    return sizeof(AnisoSVOctreeCPU) + levelRanges.size() * sizeof(Vector2ui);
}