#pragma once

#include <cstdint>

#include "RayLib/Ray.h"
#include "RayLib/AABB.h"
#include "RayLib/TracerStructs.h"
#include "RayLib/CoordinateConversion.h"

#include "GPULightI.h"
#include "DeviceMemory.h"
#include "MortonCode.cuh"

struct PathGuidingNode;
class CudaSystem;

class AnisoSVOctreeGPU
{
    private:
    template <class T>
    struct AnisoData
    {
        Vector<4, T> data[2];

        __device__ void AtomicAdd(uint8_t index, T value);
        __device__ T    Read(uint8_t index) const;
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
    static_assert(sizeof(uint64_t) * BYTE_BITS == (IS_LEAF_BIT_COUNT +
                                                   PARENT_BIT_COUNT +
                                                   CHILD_BIT_COUNT +
                                                   CHILD_MASK_BIT_COUNT),
                  "SVO Packed Bits exceeds 64-bit uint");

    public:
    //
    static constexpr uint64_t   INVALID_NODE = 0x007FFFFFFFFFFFFF;
    static constexpr uint32_t   VOXEL_DIRECTION_COUNT = 8;
    // Utility Bit Options
    // Data Unpack
    __device__ static bool      IsChildrenLeaf(uint64_t packedData);
    __device__ static uint32_t  ChildMask(uint64_t packedData);
    __device__ static uint32_t  ChildrenCount(uint64_t packedData);
    __device__ static uint32_t  ChildrenIndex(uint64_t packedData);
    __device__ static uint32_t  ParentIndex(uint64_t packedData);
    // Data Pack
    __device__ static void      SetIsChildrenLeaf(uint64_t& packedData, bool);
    __device__ static void      SetChildMask(uint64_t& packedData, uint32_t);
    __device__ static void      SetChildrenIndex(uint64_t& packedData, uint32_t);
    __device__ static void      SetParentIndex(uint64_t& packedData, uint32_t);

    __device__ static void      AtomicSetChildMaskBit(uint64_t* packedData, uint32_t);
    __device__ static uint32_t  FindChildOffset(uint64_t packedData, uint32_t childId);

    __device__ static Vector3f  VoxelDirection(uint32_t directionId);
    __device__ static Vector4uc DirectionToNeigVoxels(Vector2f& interp,
                                                      const Vector3f& direction);

    private:
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
    AABB3f              svoAABB;
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
    // Uses atomics, returns false if no leaf is found on this location
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

    __device__
    Vector3f            VoxelToWorld(const Vector3ui& denseIndex);

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
    AABB3f              OctreeAABB() const;
};

class AnisoSVOctreeCPU
{
    private:
    DeviceMemory                octreeMem;
    AnisoSVOctreeGPU            treeGPU;
    // Level Ranges
    std::vector<uint32_t>       levelNodeOffsets; // Node Range of each level (except leaf)

    public:
    // Constructors & Destructor
                                AnisoSVOctreeCPU() = default;
                                ~AnisoSVOctreeCPU() = default;

    // Construct SVO using the primitives on the Accelerators
    // initialize the radiances with the Lights
    TracerError                 Constrcut(const AABB3f& sceneAABB, uint32_t resolutionXYZ,
                                          const AcceleratorBatchMap&,
                                          const GPULightI** dSceneLights,
                                          uint32_t totalLightCount,
                                          HitKey boundaryLightKey,
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

template <class T>
__device__ inline
void AnisoSVOctreeGPU::AnisoData<T>::AtomicAdd(uint8_t index, T value)
{
    uint8_t iMSB = index >> 2;
    uint8_t iLower = index & 0b11;
    atomicAdd(&(data[iMSB][iLower]), value);
}

template <class T>
__device__ inline
T AnisoSVOctreeGPU::AnisoData<T>::Read(uint8_t index) const
{
    uint8_t iMSB = index >> 2;
    uint8_t iLower = index & 0b11;
    return  data[iMSB][iLower];
}

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
    , svoAABB(Vector3f(0.0f), Vector3f(0.0f))
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
uint32_t AnisoSVOctreeGPU::ChildMask(uint64_t packedData)
{
    return static_cast<uint32_t>((packedData >> CHILD_MASK_OFFSET) & CHILD_MASK_BIT_MASK);
}

__device__ inline
uint32_t AnisoSVOctreeGPU::ChildrenCount(uint64_t packedData)
{
    return __popc(ChildMask(packedData));
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
void AnisoSVOctreeGPU::SetChildMask(uint64_t& packedData, uint32_t mask)
{
    assert(mask < (1 << CHILD_MASK_BIT_COUNT));
    static constexpr uint64_t NEGATIVE_MASK = ~(CHILD_MASK_BIT_MASK << CHILD_MASK_OFFSET);
    packedData &= NEGATIVE_MASK;
    packedData |= (static_cast<uint64_t>(mask) << CHILD_MASK_OFFSET);
}

__device__ inline
void AnisoSVOctreeGPU::AtomicSetChildMaskBit(uint64_t* packedData, uint32_t mask)
{
    assert(mask < (1 << CHILD_MASK_BIT_COUNT));
    // Atomically set the value to the mask
    atomicOr(packedData, static_cast<uint64_t>(mask) << CHILD_MASK_OFFSET);
}

__device__ inline
void AnisoSVOctreeGPU::SetChildrenIndex(uint64_t& packedData, uint32_t childIndex)
{
    assert(childIndex < (1 << CHILD_BIT_COUNT));
    static constexpr uint64_t NEGATIVE_MASK = ~(CHILD_BIT_MASK << CHILD_OFFSET);
    packedData &= NEGATIVE_MASK;
    packedData |= (static_cast<uint64_t>(childIndex) << CHILD_OFFSET);
}

__device__ inline
void AnisoSVOctreeGPU::SetParentIndex(uint64_t& packedData, uint32_t parentIndex)
{
    assert(parentIndex < (1 << PARENT_BIT_COUNT));
    static constexpr uint64_t NEGATIVE_MASK = ~(PARENT_BIT_MASK << PARENT_OFFSET);
    packedData &= NEGATIVE_MASK;
    packedData |= (static_cast<uint64_t>(parentIndex) << PARENT_OFFSET);
}

__device__ inline
uint32_t AnisoSVOctreeGPU::FindChildOffset(uint64_t packedData, uint32_t childId)
{
    assert(childId < 8);
    uint32_t mask = ChildMask(packedData);
    uint32_t bitLoc = (1 << static_cast<uint32_t>(childId));
    assert((bitLoc & mask) != 0);
    uint32_t lsbMask = bitLoc - 1;
    uint32_t offset = __popc(lsbMask & mask);
    assert(offset < 8);
    return offset;
}

__device__ inline
Vector3f AnisoSVOctreeGPU::VoxelDirection(uint32_t directionId)
{
    static constexpr Vector3f X_AXIS = XAxis;
    static constexpr Vector3f Y_AXIS = YAxis;
    static constexpr Vector3f Z_AXIS = ZAxis;

    int8_t signX = (directionId >> 0) & 0b1;
    int8_t signY = (directionId >> 1) & 0b1;
    int8_t signZ = (directionId >> 2) & 0b1;
    signX = (1 - signX) * 2 - 1;
    signY = (1 - signY) * 2 - 1;
    signZ = (1 - signZ) * 2 - 1;

    Vector3f dir = (X_AXIS * signX +
                    Y_AXIS * signY +
                    Z_AXIS * signZ);
    return dir.Normalize();
}

__device__ inline
Vector4uc AnisoSVOctreeGPU::DirectionToNeigVoxels(Vector2f& interp,
                                                  const Vector3f& direction)
{
    // I couldn't comprehend this as a mathematical
    // representation so tabulated the output
    static constexpr Vector4uc TABULATED_LAYOUTS[12] =
    {
        Vector4uc(3,2,0,1), Vector4uc(0,3,1,2),  Vector4uc(0,1,2,3), Vector4uc(2,1,3,0),
        Vector4uc(0,1,4,5), Vector4uc(1,2,5,6),  Vector4uc(2,3,6,7), Vector4uc(3,0,7,4),
        Vector4uc(4,5,7,6), Vector4uc(5,6,4,7),  Vector4uc(6,7,5,4), Vector4uc(7,4,6,5)
    };

    static constexpr float PIXEL_X = 4;
    static constexpr float PIXEL_Y = 2;

    Vector2f thetaPhi = Utility::CartesianToSphericalUnit(direction);
    // Normalize to generate UV [0, 1]
    // theta range [-pi, pi]
    float u = (thetaPhi[0] + MathConstants::Pi) * 0.5f / MathConstants::Pi;
    // phi range [0, pi]
    float v = 1.0f - (thetaPhi[1] / MathConstants::Pi);

    // Convert to pixelCoords
    float pixelX = u * PIXEL_X;
    float pixelY = v * PIXEL_Y;

    float indexX;
    float interpX = modff(pixelX + 0.5f, &indexX);
    indexX -= 1.0f;
    uint32_t indexXInt = signbit(indexX) ? 3 : static_cast<uint32_t>(indexX);

    float indexY;
    float interpY = abs(modff(pixelY + 0.5f, &indexY));
    uint32_t indexYInt = static_cast<uint32_t>(indexX);

    interp = Vector2f(interpX, interpY);
    return TABULATED_LAYOUTS[indexYInt * 4 + indexXInt];
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
    if(leafFound)
    {
        // Extrapolate the data to the all appropriate locations
        Vector2f interpValues;
        Vector4uc neighbours = DirectionToNeigVoxels(interpValues,
                                                     outgoingDir);
        // Deposition should be done in a
        // Box filter like fashion
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            dLeafRadianceWrite[lIndex].AtomicAdd(neighbours[i], radiance);
            dLeafSampleCountWrite[lIndex].AtomicAdd(neighbours[i], 1);
        }
    }
    return leafFound;
}

__device__ inline
bool AnisoSVOctreeGPU::LeafIndex(uint32_t& index, const Vector3f& worldPos) const
{
    // Useful constants
    static constexpr uint32_t DIMENSION = 3;
    static constexpr uint32_t DIM_MASK = (1 << DIMENSION) - 1;

    if(svoAABB.IsOutside(worldPos))
    {
        index = UINT32_MAX;
        return false;
    }

    // Calculate Dense Voxel Id
    Vector3f lIndex = ((worldPos - svoAABB.Min()) / leafVoxelSize).Floor();
    Vector3ui denseIndex = Vector3ui(lIndex[0], lIndex[1], lIndex[2]);
    // Generate Morton code of the index
    uint64_t voxelMorton = MortonCode::Compose<uint64_t>(denseIndex);
    uint32_t mortonLevelShift = (leafDepth - 1) * DIMENSION;
    // Now descend down
    uint32_t currentNodeIndex = 0;
    for(uint32_t i = 0; i < leafDepth; i++)
    {
        uint64_t currentNode = dNodes[currentNodeIndex];
        uint32_t childId = (voxelMorton >> mortonLevelShift) & DIM_MASK;
        // Check if this node has that child avail
        if(!((ChildMask(currentNode) >> childId) & 0b1))
        {
            //printf("Child NF\n");
            index = UINT32_MAX;
            return false;
        }
        uint32_t childOffset = FindChildOffset(currentNode, childId);
        uint32_t childrenIndex = ChildrenIndex(currentNode);
        currentNodeIndex = childrenIndex + childOffset;
        mortonLevelShift -= DIMENSION;
    }
    // If we descended down properly currentNode should point to the
    // index. Notice that, last level's children ptr will point to the leaf arrays
    index = currentNodeIndex;
    return true;
}

__device__ inline
uint32_t AnisoSVOctreeGPU::FindBin(bool& isLeaf, uint32_t upperLimit, uint32_t leafIndex) const
{
    return UINT32_MAX;
}

__device__ inline
Vector3f AnisoSVOctreeGPU::VoxelToWorld(const Vector3ui& denseIndex)
{
    Vector3f denseIFloat = Vector3f(static_cast<float>(denseIndex[0]) + 0.5f,
                                    static_cast<float>(denseIndex[1]) + 0.5f,
                                    static_cast<float>(denseIndex[2]) + 0.5f);
    return svoAABB.Min() + (denseIFloat * leafVoxelSize);
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
AABB3f AnisoSVOctreeGPU::OctreeAABB() const
{
    return svoAABB;
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
    return sizeof(AnisoSVOctreeCPU) + levelNodeOffsets.size() * sizeof(Vector2ui);
}