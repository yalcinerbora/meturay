#pragma once

#include <cstdint>

#include "RayLib/Ray.h"
#include "RayLib/AABB.h"
#include "RayLib/TracerStructs.h"
#include "RayLib/CoordinateConversion.h"
#include "RayLib/HybridFunctions.h"
#include "RayLib/MortonCode.h"
#include "RayLib/ColorConversion.h"


#include "GPULightI.h"
#include "DeviceMemory.h"
#include "BinarySearch.cuh"
#include "GaussianLobe.cuh"

struct WFPGPathNode;
class CudaSystem;

// Voxel Payload (SoA)
struct VoxelPayload
{
    private:
    static constexpr uint32_t NORMAL_X_BIT_COUNT        = 9;
    static constexpr uint32_t NORMAL_Y_BIT_COUNT        = 9;
    static constexpr uint32_t NORMAL_LENGTH_BIT_COUNT   = 7;
    static constexpr uint32_t SPECULAR_BIT_COUNT        = 6;
    static constexpr uint32_t NORMAL_SIGN_BIT_COUNT     = 1;

    static_assert((NORMAL_X_BIT_COUNT +
                   NORMAL_Y_BIT_COUNT +
                   NORMAL_LENGTH_BIT_COUNT +
                   SPECULAR_BIT_COUNT +
                   NORMAL_SIGN_BIT_COUNT) == sizeof(uint32_t) * BYTE_BITS);

    static constexpr uint32_t NORMAL_X_BIT_MASK         = (1 << NORMAL_X_BIT_COUNT) - 1;
    static constexpr uint32_t NORMAL_Y_BIT_MASK         = (1 << NORMAL_Y_BIT_COUNT) - 1;
    static constexpr uint32_t NORMAL_LENGTH_BIT_MASK    = (1 << NORMAL_LENGTH_BIT_COUNT) - 1;
    static constexpr uint32_t SPECULAR_BIT_MASK         = (1 << SPECULAR_BIT_COUNT) - 1;
    static constexpr uint32_t NORMAL_SIGN_BIT_MASK      = (1 << NORMAL_SIGN_BIT_COUNT) - 1;

    static constexpr float UNORM_SPEC_FACTOR            = 1.0f / static_cast<float>(SPECULAR_BIT_COUNT);
    static constexpr float UNORM_LENGTH_FACTOR          = 1.0f / static_cast<float>(NORMAL_LENGTH_BIT_MASK);
    static constexpr float UNORM_NORM_X_FACTOR          = 1.0f / static_cast<float>(NORMAL_X_BIT_MASK);
    static constexpr float UNORM_NORM_Y_FACTOR          = 1.0f / static_cast<float>(NORMAL_Y_BIT_MASK);

    static constexpr uint32_t NORMAL_X_OFFSET           = 0;
    static constexpr uint32_t NORMAL_Y_OFFSET           = NORMAL_X_OFFSET + NORMAL_X_BIT_COUNT;
    static constexpr uint32_t NORMAL_LENGTH_OFFSET      = NORMAL_Y_OFFSET + NORMAL_Y_BIT_COUNT;
    static constexpr uint32_t SPECULAR_OFFSET           = NORMAL_LENGTH_OFFSET + NORMAL_LENGTH_BIT_COUNT;
    static constexpr uint32_t NORMAL_SIGN_BIT_OFFSET    = SPECULAR_OFFSET + SPECULAR_BIT_COUNT;

    public:
    // Leaf Unique Data
    Vector2f*   dTotalIrradianceLeaf;   // Total Irradiance, 2 values
                                        // one for "above" the surface other is for "below".
                                        // Surface is defined by the normal
    Vector2ui*  dSampleCountLeaf;       // Total number of ray samples that hits to this voxel
                                        // used to calculate average irradiance
    // Leaf Common Data
    Vector2h*   dAvgIrradianceLeaf;     // Average irradiance of this node, this value is also filtered
                                        // and contains information from nearby voxels
    uint32_t*   dNormalAndSpecLeaf;     // Packed data; contains normal, and guiding metric
                                        // 32-bit word => [ 9 | 9 | 6 | 8 ]
                                        // first two contains normal X, Y coords (9-bit UNORM in DirectX terms),
                                        // third field hold normal std deviation (since SVO is low resolution
                                        // discretization normals are not exact (average and distribution).
                                        // Last field holds the specularity of the location; again it is
                                        // normalized 8-bit integer, 0 means fully diffuse and 1 means
                                        // perfectly specular.
    uint8_t*    dGuidingFactorLeaf;     // Guiding metric which is used to determine if this location
                                        // of the scene should be guided by the algorithm or not.
                                        // this value is used stochastically. It is UNORM-8.
    uint64_t*   dMicroQuadTreeLeaf;     // Micro partition tree of the irradiance

    // Node Data
    Vector2h*   dAvgIrradianceNode;     // Same as above but for nodes
    uint32_t*   dNormalAndSpecNode;
    uint8_t*    dGuidingFactorNode;
    uint64_t*   dMicroQuadTreeNode;

    // Read Routines
    __device__
    Vector3f            ReadNormalAndSpecular(float& stdDev, float& specularity,
                                              uint32_t nodeIndex, bool isLeaf) const;

    __device__
    float               ReadRadiance(const Vector3f& coneDirection, float coneAperture,
                                     uint32_t nodeIndex, bool isLeaf) const;
    __device__
    float               ReadGuidingFactor(uint32_t nodeIndex, bool isLeaf) const;

    __device__
    bool                WriteLeafRadiance(const Vector3f& wi, const Vector3f& wo,
                                          const Vector3f& surfaceNormal,
                                          uint32_t leafIndex, float radiance);
    __device__
    void                WriteNormalAndSpecular(const Vector3f& normal, float specularity,
                                               uint32_t nodeIndex, bool isLeaf);

    // Size Related (per voxel and total)
    static size_t       BytePerLeaf();
    static size_t       BytePerNode();
    static size_t       TotalSize(size_t leafCount, size_t nodeCount);
};

class AnisoSVOctreeGPU
{
    public:
    static constexpr int VOXEL_DIR_DATA_COUNT = 8;

    private:
    static constexpr uint16_t LAST_BIT_UINT16 = (sizeof(uint16_t) * BYTE_BITS - 1);

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

    static constexpr uint32_t INVALID_PARENT        = PARENT_BIT_MASK;

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
    __device__ static bool      HasChild(uint64_t packedData, uint32_t childId);

    // Bin info related packing operations
    __device__ static bool      IsBinMarked(uint16_t binInfo);
    __device__ static void      SetBinAsMarked(uint16_t& gNodeBinInfo);
    __device__ static uint16_t  GetRayCount(uint16_t binInfo);

    private:
    // Generic Data
    uint32_t*           dLevelNodeOffsets;  // Nodes (except leafs which have different array)
                                            // are laid out on a single array in a depth first fasion
                                            // this offset can be used to determine which level
                                            // of nodes start-end on this array
                                            // This data is convenience for level by level
                                            // GPU kernel calls
    // SVO Data
    uint64_t*           dNodes;             // Node structure, unlike other SVO structures
                                            // nodes hold parent pointer as well for stackless traversal
                                            // children ptr (28) parent ptr (27), child mask (8), leafBit (1)
    uint16_t*           dBinInfo;           // Number of rays that "collide" with this voxel
                                            // similar positioned rays will collaborate to generate
                                            // an incoming radiance field to path guide
                                            // MSB is "isCollapsed" bit which means that
                                            // there are not enough rays on this voxels
                                            // collaborating rays should use the upper level
    // Leaf Data (Leafs does not have nodes,
    // if leaf bit is set on parent children ptr points to these)
    uint32_t*           dLeafParents;       // Parent pointers of the leaf it is 32-bit for padding reasons
                                            // (nodes have 27-bit parent pointers so latter 5-bits are wasted)
    uint16_t*           dLeafBinInfo;       // Same as above but for leafs

    // Voxel Payload (refer to the payload structure for info)
    // TODO: Make this a template for generic SVO structure in future
    VoxelPayload        payload;

    // Boundary Light (Used when any ray does not hit anything)
    // Only single boundary light is supported currently which should be enough for many
    // scenes and demonstration purposes
    const GPULightI*    dBoundaryLight;
    // Constants
    AABB3f              svoAABB;            // svoAABB is slightly larger than the scene AABB;
                                            // additionally it is a cube
                                            // to prevent edge voxelization
    uint32_t            voxelResolution;    // x = y = z
    uint32_t            leafDepth;          // log2(voxelResolution)
    uint32_t            nodeCount;          // Total number of nodes in the SVO
    uint32_t            leafCount;          // Total number of leaves in the SVO
    float               leafVoxelSize;      // svoAABB / voxelResolution
    uint32_t            levelOffsetCount;   // Number of data on the "dLevelNodeOffsets" array
    // CPU class can access and set the data
    friend class        AnisoSVOctreeCPU;

    public:
    // Constructors & Destructor
    __host__            AnisoSVOctreeGPU();
    // Methods
    // Trace the ray over the SVO; returns tMin and leafId/nodeId
    // cone can terminate on a node due to its aperture
    __device__
    float               ConeTraceRay(bool& isLeaf, uint32_t& nodeId, const RayF&,
                                     float tMin, float tMax, float coneAperture = 0.0f,
                                     uint32_t maxQueryLevel = 0) const;

    // Deposit radiance to the nearest voxel leaf
    // Uses atomics, returns false if no leaf is found on this location
    __device__
    bool                DepositRadiance(const Vector3f& worldPos,
                                        const Vector3f& surfaceNormal,
                                        const Vector3f& wi,
                                        const Vector3f& wo,
                                        float radiance);

    // Read the radiance value from the specified node,
    // converts irradiance to radiance
    // Result will be the bi-linear spherical interpolation of
    // nearest samples
    __device__
    float                ReadRadiance(const Vector3f& coneDirection, float coneAperture,
                                     uint32_t nodeId, bool isLeaf) const;
    __device__
    Vector3f            DebugReadNormal(float& stdDev, uint32_t nodeId, bool isLeaf) const;

    // Atomically Increment the ray count for that leafIndex
    __device__
    void                IncrementLeafRayCount(uint32_t leafIndex);

    __device__
    bool                Descend(uint32_t& index, uint64_t mortonCode, uint32_t depth) const;
    // Return the leaf index of this position
    // Return false if no such leaf exists
    // Due to numerical precision, "worldPos" may be slightly outside of a voxel
    // user can set "checkNeighbours" to find the closest neighboring voxel
    __device__
    bool                NearestNodeIndex(uint32_t& index, const Vector3f& worldPos,
                                         uint32_t depth,
                                         bool checkNeighbours = false) const;
    // Find the bin from the leaf
    // Bin is the node that is the highest non-collapsed node
    __device__
    uint32_t            FindMarkedBin(bool& isLeaf, uint32_t initialLeafIndex)  const;
    // Returns the center of the voxel as world space coordinates
    __device__
    Vector3f            VoxelToWorld(const Vector3ui& denseIndex) const;
    // Returns the voxel size of the specified node index
    // This operation binary search the nodeIndex with the "dLevelNodeOffsets"
    // array, thus, it can be slow (due to global memory access)
    __device__
    float               NodeVoxelSize(uint32_t nodeIndex, bool isLeaf) const;
    // Find out the voxel center position directly from the node
    __device__
    Vector3f            NodeVoxelPosition(uint32_t nodeIndex, bool isLeaf) const;

    // Direct Set Functions
    // Normal and light emitted radiance information will be set by these
    // on initialization phase
    __device__
    bool                SetLeafRadiance(uint64_t mortonCode,
                                        const Vector2f& combinedLuminance,
                                        const Vector2ui& initialSampleCount);
    __device__
    bool                SetLeafNormal(uint64_t mortonCode, Vector3f combinedNormal);

    // Accessors
    __device__
    float               LeafVoxelSize() const;
    __device__
    uint32_t            LeafCount() const;
    __device__
    uint32_t            LeafDepth() const;
    __device__
    uint32_t            NodeCount() const;
    __device__
    uint32_t            VoxelResolution() const;
    __device__
    AABB3f              OctreeAABB() const;

    // Convenience Functions
    __device__
    static uint16_t     AtomicAddUInt16(uint16_t* gLocation, uint16_t value);
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
    // Construct (Actually Load) the tree directly from binary
    TracerError                 Constrcut(const std::vector<Byte>& data,
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
    void                    AccumulateRaidances(const WFPGPathNode* dPGNodes,
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

    void                    DumpSVOAsBinary(std::vector<Byte>& data,
                                            const CudaSystem& system) const;
};

__device__ inline
Vector3f VoxelPayload::ReadNormalAndSpecular(float& stdDev, float& specularity,
                                             uint32_t nodeIndex, bool isLeaf) const
{
    const uint32_t* gFetchArray = (isLeaf) ? dNormalAndSpecLeaf : dNormalAndSpecNode;
    uint32_t packedData = gFetchArray[nodeIndex];

    Vector3f normal;
    // [0, 511]
    normal[0] = static_cast<float>((packedData >> NORMAL_X_OFFSET) & NORMAL_X_BIT_MASK);
    normal[1] = static_cast<float>((packedData >> NORMAL_Y_OFFSET) & NORMAL_Y_BIT_MASK);
    // [0, 2]
    normal[0] *= UNORM_NORM_X_FACTOR * 2.0f;
    normal[1] *= UNORM_NORM_Y_FACTOR * 2.0f;
    // [-1, 1]
    normal[0] -= 1.0f;
    normal[1] -= 1.0f;
    // Z axis from normalization
    normal[2] = sqrtf(max(0.0f, 1.0f - normal[0] * normal[0] - normal[1] * normal[1]));
    normal.NormalizeSelf();

    bool normalZSign = (packedData >> NORMAL_SIGN_BIT_OFFSET) & NORMAL_SIGN_BIT_MASK;
    normal[2] *= (normalZSign) ? -1.0f : 1.0f;

    float normalLength = static_cast<float>((packedData >> NORMAL_LENGTH_OFFSET) & NORMAL_LENGTH_BIT_MASK);
    normalLength *= UNORM_LENGTH_FACTOR;

    specularity = static_cast<float>((packedData >> SPECULAR_OFFSET) & SPECULAR_BIT_MASK);
    specularity *= UNORM_SPEC_FACTOR;

    // Toksvig 2004
    stdDev = (1.0f - normalLength) / normalLength;

    return normal;
}

__device__ inline
float VoxelPayload::ReadRadiance(const Vector3f& coneDirection,
                                 float coneAperture,
                                 uint32_t nodeIndex, bool isLeaf) const
{
    const Vector2h* gIrradArray = (isLeaf) ? dAvgIrradianceLeaf : dAvgIrradianceNode;

    float normalDeviation, specularity;
    // Fetch normal to estimate the surface confidence
    Vector3f normal = ReadNormalAndSpecular(normalDeviation, specularity,
                                            nodeIndex, isLeaf);

    // TODO: Implement radiance distribution incorporation
    // Generate a Gaussian using the bit quadtree for outgoing radiance
    // Incorporate normal, (which also is a gaussian)
    // Convolve all these three to generate an analytic function (freq domain)
    // All functions are gaussians so fourier transform is analytic
    // Convert it back
    // Sample the value using tetha (coneDir o normal)

    //// TODO: Change these to somewhat proper values
    //static constexpr float MAX_SOLID_ANGLE = 4.0f * MathConstants::Pi;
    //static constexpr float INV_MAX_SOLID_ANGLE = 1.0f / MAX_SOLID_ANGLE;
    //static constexpr float SHARPNESS_EXPANSION = 100.0f;
    //float solidDeviation = 1.0f - ((MAX_SOLID_ANGLE - coneAperture) * INV_MAX_SOLID_ANGLE);

    //// Generate Gaussian Lobe for cone
    //Vector2h irrads = gIrradArray[nodeIndex];
    //GaussianLobe normalLobe0(normal, irrads[0], 0.01);// normalDeviation* SHARPNESS_EXPANSION);
    //GaussianLobe normalLobe1(-normal, irrads[1], 0.01);// normalDeviation* SHARPNESS_EXPANSION);
    //GaussianLobe coneLobe(-coneDirection, 1.0f, 0.01);// solidDeviation* SHARPNESS_EXPANSION);

    //float totalIrrad = max(0.0f, normalLobe0.Dot(coneLobe)) + max(0.0f, normalLobe1.Dot(coneLobe));
    //return totalIrrad;

    // Currently we are just fetching irradiance
    // which side are we on
    // Note that "coneDirection" is towards to the surface
    bool towardsNormal = (normal.Dot(-coneDirection) >= 0.0f);
    uint32_t index = towardsNormal ? 0 : 1;
    float result = gIrradArray[nodeIndex][index];
    return result;// *abs(normal.Dot(-coneDirection));
}

__device__ inline
float VoxelPayload::ReadGuidingFactor(uint32_t nodeIndex, bool isLeaf) const
{
    const uint8_t* gGuidingFactorArray = (isLeaf) ? dGuidingFactorLeaf : dGuidingFactorNode;
    return gGuidingFactorArray[nodeIndex];
}

__device__ inline
bool VoxelPayload::WriteLeafRadiance(const Vector3f& surfaceNormal,
                                     const Vector3f& wi,
                                     const Vector3f& wo,
                                     uint32_t leafIndex,
                                     float radiance)
{
    float normalDeviation, specularity;
    // Fetch normal to estimate the surface confidence
    Vector3f normal = ReadNormalAndSpecular(normalDeviation, specularity,
                                            leafIndex, true);

    // Use actual surface normal to determine which side
    // we should accumulate
    float NdN = normal.Dot(surfaceNormal);
    float NdL = normal.Dot(wo);
    float NdV = normal.Dot(wi);

    bool towardsFront = ((NdN >= 0.0f)
                         && (NdL >= 0.0f)
                         //&& (NdV >= 0.0f)
                         );

    bool towardsBack = ((NdN < 0.0f)
                        && (NdL < 0.0f)
                        //&& (NdV < 0.0f)
                        );

    if(!towardsFront && !towardsBack)
        return true;

    uint32_t index = towardsFront ? 0 : 1;

    // TODO: Distribute the radiance properly using normals std deviation
    //float cosAlpha = wo.Dot(normal);
    //float cosBeta = wi.Dot(normal);
    //bool towardsNormal = (cosAlpha >= 0.0f);/* && (cosBeta >= 0.0f);*/
    //bool towardsBackNormal = (cosAlpha < 0.0f) && (cosBeta < 0.0f);

    //// If this vertex represents inter-reflection
    //// Don't accumulate (this will create light leak)
    //if(!towardsNormal && !towardsBackNormal)
    //    return true;

    // Atomic operation here since many rays may update on learn operation
    atomicAdd(&(dTotalIrradianceLeaf[leafIndex][index]), radiance);
    atomicAdd(&(dSampleCountLeaf[leafIndex][index]), 1);
}

__device__ inline
void VoxelPayload::WriteNormalAndSpecular(const Vector3f& normal, float specularity,
                                          uint32_t nodeIndex, bool isLeaf)
{
    uint32_t* gWriteArray = (isLeaf) ? dNormalAndSpecLeaf : dNormalAndSpecNode;

    float length = normal.Length();
    // NormalXY [-1, 1]
    Vector2f normalXY = normal * Vector3f(1.0f / length);
    // NormalXY [0, 2]
    normalXY += Vector2f(1.0f);
    // NormalXY [0, 1]
    normalXY *= Vector2f(0.5f);
    // NormalXY [0, 2^9)
    normalXY *= Vector2f(NORMAL_X_BIT_MASK, NORMAL_Y_BIT_MASK);
    Vector2ui packedNorm = Vector2ui(normalXY);
    // Length [0, 1]
    // Length [0, 2^7)
    length *= static_cast<float>(NORMAL_LENGTH_BIT_MASK);
    uint32_t packedLength = static_cast<uint32_t>(length);
    // Specular [0, 1]
    // Specular [0, 2^6)
    specularity *= static_cast<float>(SPECULAR_BIT_MASK);
    uint32_t packedSpecular = static_cast<uint32_t>(specularity);

    uint32_t packedZSign = signbit(normal[2]) ? 1 : 0;

    uint32_t packedData = 0;
    packedData |= packedNorm[0] << NORMAL_X_OFFSET;
    packedData |= packedNorm[1] << NORMAL_Y_OFFSET;
    packedData |= packedLength << NORMAL_LENGTH_OFFSET;
    packedData |= packedSpecular << SPECULAR_OFFSET;
    packedData |= packedZSign << NORMAL_SIGN_BIT_OFFSET;

    gWriteArray[nodeIndex] = packedData;
}

inline
size_t VoxelPayload::BytePerLeaf()
{
    return (sizeof(Vector2f) + sizeof(Vector2ui) +
            sizeof(Vector2h) + sizeof(uint32_t) +
            sizeof(uint8_t) + sizeof(uint64_t));
}

inline
size_t VoxelPayload::BytePerNode()
{
    return (sizeof(Vector2h) + sizeof(uint32_t) +
            sizeof(uint8_t) + sizeof(uint64_t));
}

inline
size_t VoxelPayload::TotalSize(size_t leafCount, size_t nodeCount)
{
    return (BytePerLeaf() * leafCount +
            BytePerNode() * nodeCount);
}

__device__ inline
uint16_t AnisoSVOctreeGPU::AtomicAddUInt16(uint16_t* location, uint16_t value)
{
    // Atomic add of 16-bit integer using CAS Atomics
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
    uint16_t assumed;
    uint16_t old = *location;
    do
    {
        assumed = old;
        // Actual Operation
        uint16_t result = assumed + value;
        old = atomicCAS(location, assumed, result);
    }
    while(assumed != old);
    return old;
}

__host__ inline
AnisoSVOctreeGPU::AnisoSVOctreeGPU()
    : dLevelNodeOffsets(nullptr)
    , dNodes(nullptr)
    , dBinInfo(nullptr)
    , dLeafParents(nullptr)
    , dLeafBinInfo(nullptr)
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

    // Clang complains this function args does not match
    static_assert(sizeof(uint64_t) == sizeof(unsigned long long));

    // Atomically set the value to the mask
    atomicOr(reinterpret_cast<unsigned long long*>(packedData),
             static_cast<unsigned long long>(mask) << CHILD_MASK_OFFSET);
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
bool AnisoSVOctreeGPU::HasChild(uint64_t packedData, uint32_t childId)
{
    assert(childId < 8);
    uint32_t childMask = ChildMask(packedData);
    return (childMask >> childId) & 0b1;
}

__device__ inline
bool AnisoSVOctreeGPU::IsBinMarked(uint16_t rayCounts)
{
    return (rayCounts >> LAST_BIT_UINT16) & 0b1;
}

__device__ inline
void AnisoSVOctreeGPU::SetBinAsMarked(uint16_t& gNodeRayCount)
{
    uint16_t expandedBoolean = (1u << LAST_BIT_UINT16);
    gNodeRayCount |= expandedBoolean;
}

__device__ inline
uint16_t AnisoSVOctreeGPU::GetRayCount(uint16_t binInfo)
{
    return binInfo & ((1u << LAST_BIT_UINT16) - 1);
}

__device__ inline
float AnisoSVOctreeGPU::ConeTraceRay(bool& isLeaf, uint32_t& leafId, const RayF& ray,
                                     float tMin, float tMax,  float coneAperture,
                                     uint32_t maxQueryLevelOffset) const
{
    static constexpr float EPSILON = MathConstants::Epsilon;
    // We wrap the exp2f function here
    // since you can implement an integer version
    // with some bit manipulation
    // It is maybe faster?
    auto FastNegExp2f = [](int power)
    {
        // Just manipulate exponent bits
        static constexpr int IEEE754_EXPONENT_OFFSET = 23;
        static constexpr int IEEE754_EXPONENT_BIAS = 127;
        // This expression probably make it slow tho
        // 2^0 does not fit the mathematical expression
        // We need a special case here
        if(power == 0) return 1.0f;
        uint32_t fVal = IEEE754_EXPONENT_BIAS - power;
        fVal <<= IEEE754_EXPONENT_OFFSET;
        return __int_as_float(fVal);

        // This is the float version
        //return exp2f(-power);
    };
    // Instead of holding a large stack for each thread,
    // we hold a bit stack which will hold the node morton Id
    // when we "pop" from the stack we will use global parent pointer
    // to pop up a level then use the extracted bits (last 3 bits)
    // to find the voxel corner position
    //
    // An example
    // Root corner is implicitly (0, 0) (2D example)
    // If morton stack has 11
    // corner = 0 + exp2f(-(morton3BitCount + 1)) if(axis bit is 1 else 0)
    // for each dimension
    // if we pop up a level from (0.5, 0.5) and the bit are 11
    // p = 0.5 - exp2f(-(morton3BitCount + 1))  if(axis bit is 1 else 0)
    uint64_t mortonBitStack = 0;
    uint8_t stack3BitCount = 0;
    // Helper Lambdas for readability (Basic stack functionality over the variables)
    auto PushMortonCode = [&mortonBitStack, &stack3BitCount](uint32_t bitCode)
    {
        assert(bitCode < 8);
        mortonBitStack = (mortonBitStack << 3) | bitCode;
        stack3BitCount += 1;
        assert(stack3BitCount <= 21);
    };
    auto PopMortonCode = [&mortonBitStack, &stack3BitCount]()
    {
        mortonBitStack >>= 3;
        stack3BitCount -= 1;
    };
    auto ReadMortonCode = [&mortonBitStack]() -> uint32_t
    {
        return mortonBitStack & 0b111;
    };

    // Aperture Related Routines
    const float CONE_DIAMETER_FACTOR = tan(0.5f * coneAperture) * 2.0f;
    auto ConeDiameter = [CONE_DIAMETER_FACTOR](float distance)
    {
        return CONE_DIAMETER_FACTOR * distance;
    };
    auto LevelVoxelSize = [this](uint32_t currentLevel)
    {
        float levelFactor = static_cast<float>(1 << (leafDepth - currentLevel));
        return levelFactor * leafVoxelSize;
    };

    // Set leaf to invalid value
    leafId = UINT32_MAX;

    // Pull ray to the registers
    Vector3f rDir = ray.getDirection();
    Vector3f rPos = ray.getPosition();
    // On AABB intersection tests (Voxels are special case AABBs)
    // we will use 1 / rayDirection in their calculations
    // If any of the directions are parallel to any of the axes
    // t values will explode (or NaN may happen (0 / 0))
    // In order to prevent that use an epsilon here
    if(fabs(rDir[0]) < EPSILON) rDir[0] = copysignf(EPSILON, rDir[0]);
    if(fabs(rDir[1]) < EPSILON) rDir[1] = copysignf(EPSILON, rDir[1]);
    if(fabs(rDir[2]) < EPSILON) rDir[2] = copysignf(EPSILON, rDir[2]);

    // Ray is in world space convert to SVO space
    // (SVO is in [0,1])
    // Translate the ray to -AABBmin
    // Scale the ray from  [AABBmin, AABBmax] to [0,1]
    const Vector3f svoTranslate = -svoAABB.Min();
    const Vector3f svoScale = Vector3f(1.0f) / (svoAABB.Max() - svoAABB.Min());
    rDir = rDir * svoScale;
    rPos = (rPos + svoTranslate) * svoScale;
    // Now voxel span calculations become trivial
    // 1.0f, 0.5f, 0.25f ...
    // You can use integer exp2f version now

    // Now also align all the rays as if their direction is positive
    // on all axes and we will
    // mirror the SVO appropriately
    rPos[0] = (rDir[0] < 0.0f) ? (1.0f - rPos[0]) : rPos[0];
    rPos[1] = (rDir[1] < 0.0f) ? (1.0f - rPos[1]) : rPos[1];
    rPos[2] = (rDir[2] < 0.0f) ? (1.0f - rPos[2]) : rPos[2];
    // Mask will flip the childId to mirrored childId
    // XOR the child bit mask with this value when traversing
    uint8_t mirrorMask = 0b000;
    if(rDir[0] < 0.0f) mirrorMask |= 0b001;
    if(rDir[1] < 0.0f) mirrorMask |= 0b010;
    if(rDir[2] < 0.0f) mirrorMask |= 0b100;

    // Generate Direction Coefficient (All ray directions are positive now
    // so abs the direction as well)
    const Vector3f dirCoeff = Vector3f(1.0f) / rDir.Abs();
    // Every AABB(Voxel) Test will have this formula
    //
    // (planes - rayPos) * dirCoeff - tCurrent
    //
    // In order to hit compiler to use mad (multiply and add) instructions,
    // expand the parenthesis and save the "rayPos * dirCoeff"
    //
    // planes * dirCoefff - rayPos - tCurrent = 0
    //
    // planes * dirCoefff - rayPos = tCurrent
    // we can check if this is greater or smaller etc.
    const Vector3f posBias = rPos * dirCoeff;

    // Since we mirrored the svo and rays are always goes through
    // as positive, we can ray march only by checking the "tMin"
    // (aka. "Bottom Left" corner)
    // We will only check tMax to find if we leaved the voxel

    // Do initial AABB check with ray's tMin
    // Only with lower left corner (first iteration will check
    // the upper right corner and may terminate)
    Vector3f tMinXYZ = (-posBias);
    // Since every ray has positive direction, "tMin" will have the
    // minimum values. Maximum of "tMin" can be used to advance the
    // ray.
    // Normally for every orientation (wrt. ray dir)
    // you need to check the top right corner as well
    float tSVO = max(max(tMinXYZ[0], tMinXYZ[1]), tMinXYZ[2]);
    // Move the point to the "bottom left" corners
    // Only do this if and of the three axes are not negative
    if(!(tMinXYZ < Vector3f(tMin)))
    {
        tMin = tSVO;
        // Nudge the tMin here
        tMin = nextafter(tMin, INFINITY);
    }
    // Traversal initial data
    // 0 index is root
    uint32_t nodeId = 0;
    // Initially corner is zero (SVO space [0,1])
    Vector3f corner = Zero3f;
    // Empty node boolean will be used to
    // create cleaner code.
    // Initially, there should be at least one node
    // available on the tree (root)
    bool emptyNode = false;
    // Ray March Loop
    while(nodeId != INVALID_PARENT)
    {
        // Children Voxel Span (derived from the level)
        float childVoxSpan = FastNegExp2f(stack3BitCount + 1);
        float currentVoxSpan = FastNegExp2f(stack3BitCount);
        // Find Center
        Vector3f voxCenter = corner + Vector3f(childVoxSpan);
        Vector3f voxTopRight = corner + Vector3f(currentVoxSpan);

        // Check early pop
        Vector3f tMaxXYZ = (voxTopRight * dirCoeff - posBias);
        float tMax = min(min(tMaxXYZ[0], tMaxXYZ[1]), tMaxXYZ[2]);

        // We will "fake" traverse this empty node by just advancing
        // the tMin
        if(emptyNode)
        {
            // Find the smallest non-negative tValue
            // Advance the tMin accordingly
            tMin = tMax;
            // Nudge the tMin here
            tMin = nextafter(tMin, INFINITY);
            uint32_t mortonBits = ReadMortonCode();
            corner -= Vector3f(((mortonBits >> 0) & 0b1) ? currentVoxSpan : 0.0f,
                               ((mortonBits >> 1) & 0b1) ? currentVoxSpan : 0.0f,
                               ((mortonBits >> 2) & 0b1) ? currentVoxSpan : 0.0f);
            PopMortonCode();
            emptyNode = false;
            continue;
        }

        // We can fetch the node now
        uint64_t node = dNodes[nodeId];

        // Actual Traversal code
        // If out of bounds directly pop to parent
        if(tMax < tMin)
        {
            // Update the corner back according to the stack
            uint32_t mortonBits = ReadMortonCode();
            corner -= Vector3f(((mortonBits >> 0) & 0b1) ? currentVoxSpan : 0.0f,
                               ((mortonBits >> 1) & 0b1) ? currentVoxSpan : 0.0f,
                               ((mortonBits >> 2) & 0b1) ? currentVoxSpan : 0.0f);
            PopMortonCode();
            // Actually pop to the parent
            nodeId = ParentIndex(node);
            emptyNode = false;
        }
        // Check the children etc.
        else
        {
            uint8_t childId = 0;
            uint8_t mirChildId = 0;
            // Find the t values of the voxel center
            // Find out the childId using that
            Vector3f tMinXYZ = (voxCenter * dirCoeff - posBias);
            if(tMinXYZ[0] < tMin) childId |= 0b001;
            if(tMinXYZ[1] < tMin) childId |= 0b010;
            if(tMinXYZ[2] < tMin) childId |= 0b100;
            // Check if this childId has actually have a child
            // Don't forget to mirror the SVO
            mirChildId = childId ^ mirrorMask;

            // If the found out voxel has children
            // Traverse down
            if(HasChild(node, mirChildId))
            {
                // Descend down
                // Only update the node if it has an actual children
                nodeId = ChildrenIndex(node) + FindChildOffset(node, mirChildId);

                // We can terminate the traversal on several conditions
                // Condition #1: This child is a leaf voxel, terminate (basic case)
                // Condition #2: Consider this voxel as leaf (due to maxQueryLevelOffset parameter)
                //               and terminate as if this node is leaf
                // Condition #3: ConeApterture is larger than the this level's voxel size,
                //               Again terminate, this voxel is assumed to cover the entire solid angle
                //               of the cone
                // Condition #3
                bool isTerminated = ConeDiameter(tMin) > LevelVoxelSize(stack3BitCount + 1);
                // Condition #1
                bool isChildLeaf = IsChildrenLeaf(node);
                // Condition #2
                uint32_t voxelLevel = stack3BitCount + 1;
                isTerminated |= (static_cast<uint32_t>(leafDepth) - voxelLevel) <= maxQueryLevelOffset;

                if(isChildLeaf || isTerminated)
                {
                    // Now return the interpolation coefficients


                    isLeaf = isChildLeaf;
                    leafId = nodeId;
                    break;
                }
            }
            // If this node does not have a child
            // "Fake" traverse down to child node
            // (don't update the node variable)
            // Set empty node for next iteration
            emptyNode = !HasChild(node, mirChildId);
            // Traverse down, update Corner
            corner += Vector3f(((childId >> 0) & 0b1) ? childVoxSpan : 0.0f,
                               ((childId >> 1) & 0b1) ? childVoxSpan : 0.0f,
                               ((childId >> 2) & 0b1) ? childVoxSpan : 0.0f);
            // Push the non-mirrored child id to the mask
            // (since corners are in mirrored coordinates)
            PushMortonCode(childId);
        }
    }
    // All Done!
    return tMin;
}

__device__ inline
float AnisoSVOctreeGPU::ReadRadiance(const Vector3f& coneDirection, float coneAperture,
                                     uint32_t nodeId, bool isLeaf) const
{
    float result;
    if(nodeId == UINT32_MAX)
    {
        Vector3f resultRGB = (dBoundaryLight) ? dBoundaryLight->Emit(-coneDirection,
                                                                     Vector3f(0.0f),
                                                                     UVSurface{},
                                                                     coneAperture)
                                              : Zero3f;
        result = Utility::RGBToLuminance(resultRGB);
    }
    else result = payload.ReadRadiance(coneDirection, coneAperture,
                                       nodeId, isLeaf);

    return result;
}

__device__ inline
Vector3f AnisoSVOctreeGPU::DebugReadNormal(float& stdDev, uint32_t nodeIndex, bool isLeaf) const
{
    float specularity;
    return payload.ReadNormalAndSpecular(stdDev, specularity,
                                         nodeIndex, isLeaf);
}

__device__ inline
bool AnisoSVOctreeGPU::DepositRadiance(const Vector3f& worldPos,
                                       const Vector3f& surfaceNormal,
                                       const Vector3f& wi,
                                       const Vector3f& wo,
                                       float radiance)
{
    // Find the leaf here
    uint32_t lIndex;
    bool leafFound = NearestNodeIndex(lIndex, worldPos, leafDepth, true);

    // We should always find a leaf with the true flag
    if(leafFound)
    {
        // Atomically set the sample
        payload.WriteLeafRadiance(wi, wo, surfaceNormal,
                                  lIndex, radiance);
    }
    return leafFound;
}

__device__ inline
void AnisoSVOctreeGPU::IncrementLeafRayCount(uint32_t leafIndex)
{
    AtomicAddUInt16(dLeafBinInfo + leafIndex, 1u);
}

__device__ inline
bool AnisoSVOctreeGPU::Descend(uint32_t& index, uint64_t mortonCode, uint32_t depth) const
{
    // Useful constants
    static constexpr uint32_t DIMENSION = 3;
    static constexpr uint32_t DIM_MASK = (1 << DIMENSION) - 1;

    uint32_t mortonLevelShift = (leafDepth - 1) * DIMENSION;
    // Now descend down
    uint32_t currentNodeIndex = 0;
    for(uint32_t i = 0; i < depth; i++)
    {
        uint64_t currentNode = dNodes[currentNodeIndex];
        uint32_t childId = (mortonCode >> mortonLevelShift) & DIM_MASK;
        // Check if this node has that child avail
        if(!((ChildMask(currentNode) >> childId) & 0b1))
            return false;

        uint32_t childOffset = FindChildOffset(currentNode, childId);
        uint32_t childrenIndex = ChildrenIndex(currentNode);
        currentNodeIndex = childrenIndex + childOffset;
        mortonLevelShift -= DIMENSION;
    }
    // If we descended down properly currentNode should point to the
    // index. Notice that, last level's children ptr will point to the leaf arrays
    index = currentNodeIndex;
    return true;
};

__device__ inline
bool AnisoSVOctreeGPU::NearestNodeIndex(uint32_t& index, const Vector3f& worldPos,
                                        uint32_t depth, bool checkNeighbours) const
{

    index = UINT32_MAX;
    if(svoAABB.IsOutside(worldPos))
    {
        index = UINT32_MAX;
        return false;
    }

    float levelVoxelSize = leafVoxelSize * static_cast<float>(1 << (leafDepth - depth));

    // Calculate Dense Voxel Id
    Vector3f lIndex = ((worldPos - svoAABB.Min()) / levelVoxelSize);
    Vector3f lIndexInt;
    Vector3f lIndexFrac = Vector3f(modff(lIndex[0], &(lIndexInt[0])),
                                   modff(lIndex[1], &(lIndexInt[1])),
                                   modff(lIndex[2], &(lIndexInt[2])));
    lIndex.FloorSelf();
    // Find movement mask
    // If that bit is set we go negative instead of positive
    Vector3ui inc = Vector3ui((lIndexFrac[0] < 0.5f) ? -1 : 1,
                              (lIndexFrac[1] < 0.5f) ? -1 : 1,
                              (lIndexFrac[2] < 0.5f) ? -1 : 1);

    // Shuffle the order as well
    // Smallest fraction (w.r.t 0.5f)
    lIndexFrac = Vector3f(0.5f) - (lIndexFrac - Vector3f(0.5f)).Abs();
    // Classic bit traversal (for loop below)
    // has XYZ axis ordering (X checked first etc.)
    // we need to shuffle the order so that the closest axis will be checked
    // first
    int minIndex0 = lIndexFrac.Min();
    int minIndex1 = Vector2f(lIndexFrac[(minIndex0 + 1) % 3],
                             lIndexFrac[(minIndex0 + 2) % 3]).Min();
    minIndex1 = (minIndex0 + minIndex1 + 1) % 3;
    int minIndex2 = 3 - minIndex0 - minIndex1;
    assert((minIndex0 != minIndex1) &&
           (minIndex1 != minIndex2));
    assert(minIndex0 >= 0 && minIndex0 < 3 &&
           minIndex1 >= 0 && minIndex1 < 3 &&
           minIndex2 >= 0 && minIndex2 < 3);

    // Initial Level index
    Vector3ui denseIndex = Vector3ui(lIndex[0], lIndex[1], lIndex[2]);

    bool found = false;
    #pragma unroll
    for(uint32_t i = 0; i < 8; i++)
    {
        // Convert if clauses to mathematical expression
        Vector3ui curIndex = Vector3ui(((i >> 0) & 0b1) ? 1 : 0,
                                       ((i >> 1) & 0b1) ? 1 : 0,
                                       ((i >> 2) & 0b1) ? 1 : 0);
        // Shuffle the iteration values for the traversal
        // Nearest nodes will be checked first
        Vector3ui curShfl;
        curShfl[minIndex0] = curIndex[0];
        curShfl[minIndex1] = curIndex[1];
        curShfl[minIndex2] = curIndex[2];

        Vector3ui voxIndex = Vector3ui(denseIndex[0] + curShfl[0] * inc[0],
                                       denseIndex[1] + curShfl[1] * inc[1],
                                       denseIndex[2] + curShfl[2] * inc[2]);

        // Generate Morton code of the index
        uint64_t voxelMorton = MortonCode::Compose3D<uint64_t>(voxIndex);
        // Traverse this morton code
        found = Descend(index, voxelMorton, depth);
        // Terminate if we are only checking a single voxel
        // or a voxel is found
        if(!checkNeighbours || found) break;
    }
    return found;
}

__device__ inline
uint32_t AnisoSVOctreeGPU::FindMarkedBin(bool& isLeaf, uint32_t initialLeafIndex) const
{
    isLeaf = true;
    uint32_t binInfo = dLeafBinInfo[initialLeafIndex];
    uint32_t parentIndex = dLeafParents[initialLeafIndex];

    // Traverse towards parent terminate when marked bin is found
    // From leaf -> root a bin should always be marked
    uint32_t nodeIndex = initialLeafIndex;
    while(parentIndex != INVALID_PARENT && !IsBinMarked(binInfo))
    {
        nodeIndex = parentIndex;
        binInfo = dBinInfo[parentIndex];
        parentIndex = ParentIndex(dNodes[parentIndex]);
        isLeaf = false;
    }
    assert(parentIndex != INVALID_PARENT);
    return nodeIndex;
}

__device__ inline
Vector3f AnisoSVOctreeGPU::VoxelToWorld(const Vector3ui& denseIndex) const
{
    Vector3f denseIFloat = Vector3f(static_cast<float>(denseIndex[0]) + 0.5f,
                                    static_cast<float>(denseIndex[1]) + 0.5f,
                                    static_cast<float>(denseIndex[2]) + 0.5f);
    return svoAABB.Min() + (denseIFloat * leafVoxelSize);
}

__device__ inline
float AnisoSVOctreeGPU::NodeVoxelSize(uint32_t nodeIndex, bool isLeaf) const
{
    using namespace GPUFunctions;
    static constexpr auto BinarySearch = GPUFunctions::BinarySearchInBetween<uint32_t>;

    if(isLeaf) return leafVoxelSize;
    // Binary search the node id from the offsets
    float levelFloat;
    [[maybe_unused]] bool found = BinarySearch(levelFloat, nodeIndex,
                                               dLevelNodeOffsets,
                                               leafDepth + 1);
    assert(found);

    uint32_t levelDiff = leafDepth - static_cast<uint32_t>(levelFloat);
    float multiplier = static_cast<float>(1 << levelDiff);
    return leafVoxelSize * multiplier;
}

__device__ inline
Vector3f AnisoSVOctreeGPU::NodeVoxelPosition(uint32_t nodeIndex, bool isLeaf) const
{
    // n is [1, sizeof(uint32_t) * BYTE_BITS)
    // "n = 0" is undefined
    // if "n > _popc(mask)" returns 0 (asserts in debug mode)
    // returns one indexed location of the set bit
    auto FindNthSet = [](uint32_t input, uint32_t n) -> uint32_t
    {
        // Not enough bits on input
        if(__popc(input) < n)
        {
            assert(__popc(input) >= n);
            return 0;
        }

        uint32_t index;
        for(int i = 1; i <= n; i++)
        {
            index = __ffs(input);
            uint32_t inverse = ~(1 << (index - 1));
            //uint32_t oldMask = input;
            input &= inverse;
            //printf("[%d]--[%d] inverse %X, oldMask %X newMask %X index %u\n",
            //       blockIdx.x, i, inverse, oldMask, input, index);
        }
        return index;
    };

    // Traverse up from the voxel and find the voxel morton code
    // Then convert it to the position
    uint64_t mortonCode = 0x0;
    uint32_t bitPtr = 0;
    static constexpr uint32_t DIMENSION = 3;

    auto PushMortonToStack = [&](uint64_t node,
                                 uint32_t nodeId,
                                 uint32_t parentId)
    {
        uint32_t childrenStart = ChildrenIndex(node);
        uint32_t bitIndex = nodeId - childrenStart;

        //printf("[%d]--bit index %u = %u, %u\n", blockIdx.x,
        //       bitIndex, childrenStart, nodeId);

        uint32_t mortonSegment = FindNthSet(ChildMask(node), bitIndex + 1) - 1;

        if(mortonSegment >= (1 << DIMENSION))
        {
            //printf("[%d]--mask 0x%X, bitindex %u, mySegment %u \n",
            //       blockIdx.x,
            //       ChildMask(node),
            //       bitIndex,
            //       mortonSegment);
            assert(mortonSegment < (1 << DIMENSION));
        }


        mortonSegment = mortonSegment << bitPtr;
        mortonCode |= mortonSegment;

        bitPtr += DIMENSION;
    };

    uint32_t nodeId = nodeIndex;
    // If leaf, data is on other array
    uint64_t node;
    if(isLeaf)
    {
        uint32_t parentId = dLeafParents[nodeId];
        node = dNodes[parentId];

        //printf("myNode %llu, nodeId %u, parentId %u\n",
        //       node, nodeId, parentId);

        PushMortonToStack(node, nodeId, parentId);
        nodeId = parentId;
    }
    else
    {
        //printf("not leaf!\n");
        node = dNodes[nodeId];
    }


    while(ParentIndex(node) != INVALID_PARENT)
    {
        uint32_t parentId = ParentIndex(node);
        node = dNodes[parentId];

        //printf("[%u] myNode %llu, nodeId %u, parentId %u\n",
        //       nodeIndex, node, nodeId, parentId);

        PushMortonToStack(node, nodeId, parentId);
        //node = dNodes[parentId];
        nodeId = parentId;
    }
    // Now calculate the voxel center using morton code
    uint32_t depth = bitPtr / DIMENSION;
    uint32_t levelDiff = leafDepth - static_cast<uint32_t>(depth);
    float multiplier = static_cast<float>(1 << levelDiff);
    float voxelSizeStart = leafVoxelSize * multiplier;
    Vector3ui position = MortonCode::Decompose3D<uint64_t>(mortonCode);
    Vector3f pos = Vector3f(position) + Vector3f(0.5f);
    // Expand to worldSpaceCoords
    pos *= voxelSizeStart;
    pos += svoAABB.Min();

    //printf("[%d]--Morton:%llu, pos[%f, %f, %f], voxelSize %f, level %u, leafVoxSize %f\n",
    //       blockIdx.x, mortonCode, pos[0], pos[1], pos[2],
    //       voxelSizeStart, depth, leafVoxelSize);

    return pos;
}

__device__ inline
bool AnisoSVOctreeGPU::SetLeafRadiance(uint64_t mortonCode,
                                       const Vector2f& combinedLuminance,
                                       const Vector2ui& initialSampleCount)
{
    Vector3ui denseIndex = MortonCode::Decompose3D<uint64_t>(mortonCode);
    Vector3f worldPos = VoxelToWorld(denseIndex);

    uint32_t leafIndex;
    bool found = NearestNodeIndex(leafIndex, worldPos, leafDepth, true);
    if(!found)
    {
        KERNEL_DEBUG_LOG("Error: SVO leaf not found!\n");
        return false;
    }
    // All fine directly add
    //
    // TODO: change this
    // Add to the all directions
    atomicAdd(&(payload.dTotalIrradianceLeaf[leafIndex][0]), combinedLuminance[0]);
    atomicAdd(&(payload.dTotalIrradianceLeaf[leafIndex][1]), combinedLuminance[1]);
    atomicAdd(&(payload.dSampleCountLeaf[leafIndex][0]), initialSampleCount[0]);
    atomicAdd(&(payload.dSampleCountLeaf[leafIndex][1]), initialSampleCount[1]);
    return true;
}

__device__ inline
bool AnisoSVOctreeGPU::SetLeafNormal(uint64_t mortonCode, Vector3f combinedNormal)
{
    Vector3ui denseIndex = MortonCode::Decompose3D<uint64_t>(mortonCode);
    Vector3f worldPos = VoxelToWorld(denseIndex);

    uint32_t leafIndex;
    bool found = NearestNodeIndex(leafIndex, worldPos, leafDepth, true);
    if(!found)
    {
        KERNEL_DEBUG_LOG("Error: SVO leaf not found!\n");
        return false;
    }

    //printf("settingLeaf %llu (%f, %f, %f) -> leaf(%u)\n", mortonCode,
    //       worldPos[0], worldPos[1], worldPos[2],
    //       leafIndex);
    // TODO: what about specularity
    // Leaf normals are considered perfect
    payload.WriteNormalAndSpecular(combinedNormal, 0.0f, leafIndex, true);
    return true;
}

__device__ inline
float AnisoSVOctreeGPU::LeafVoxelSize() const
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
uint32_t AnisoSVOctreeGPU::LeafDepth() const
{
    return leafDepth;
}

__device__ inline
uint32_t AnisoSVOctreeGPU::VoxelResolution() const
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