#pragma once

#include <cstdint>

#include "RayLib/Ray.h"
#include "RayLib/AABB.h"
#include "GPULightI.h"
#include "RayLib/TracerStructs.h"
#include "DeviceMemory.h"

struct PathGuidingNode;

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
    uint32_t            voxelResolution; // x = y = z
    uint32_t            nodeCount;
    uint32_t            leafCount;
    float               voxelSize;
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
    bool                DepositRadiance(const Vector3f& worldPos, Vector3f& outgoingDir) const;
    // Return the leaf index of this position
    // Return false if no such leaf exists
    __device__
    bool                LeafIndex(uint32_t& index, const Vector3f& worldPos);
    // Find the bin from the leaf
    // Bin is the node that is the highest non-collapsed node
    __device__
    uint32_t            FindBin(bool& isLeaf, uint32_t upperLimit, uint32_t leafIndex);

    // Accessors
    __device__
    uint32_t            VoxelSize() const;
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
                                      const GPULightI** dSceneLights);
    // Normalize and filter radiances for sampling
    void                    NormalizeAndFilterRadiance();
    // Collapse the ray counts to find optimal binning
    void                    CollapseRayCounts(uint32_t minLevel);
    // Accumulate the Emissive Radiance from the paths
    void                    AccumulateRaidances(const PathGuidingNode* dPGNodes,
                                                uint32_t totalNodeCount,
                                                uint32_t maxPathNodePerRay,
                                                const CudaSystem&);
    // Clear the ray counts for the next iteration
    void                    ClearRayCounts();
    // GPU Data Struct Access
    AnisoSVOctreeGPU        TreeGPU();
    // Misc.
    size_t                  UsedGPUMemory() const;
    size_t                  UsedCPUMemory() const;
};

__device__ inline
AnisoSVOctreeGPU::AnisoSVOctreeGPU()
    : dNodes(nullptr)
    , dBoundaryLight
{}

__device__ inline
float AnisoSVOctreeGPU::TraceRay(const RayF& ray) const
{

}