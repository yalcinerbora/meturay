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
    __device__ static bool      HasChild(uint64_t packedData, uint32_t childId);

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
    // Trace the ray over the SVO and tMin and leafId
    __device__
    float               TraceRay(uint32_t& leafId, const RayF&,
                                 float tMin, float tMax) const;
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
bool AnisoSVOctreeGPU::HasChild(uint64_t packedData, uint32_t childId)
{
    assert(childId < 8);
    uint32_t childMask = ChildMask(packedData);
    return (childMask >> childId) & 0b1;
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
float AnisoSVOctreeGPU::TraceRay(uint32_t& leafId, const RayF& ray,
                                 float tMin, float tMax) const
{
    static constexpr float EPSILON = MathConstants::Epsilon;

    // TODO:
    // We wrap the exp2f function here
    // since you can implement an integer version
    // with some bit manipulation
    // It is maybe faster?
    auto FastNegExp2f = [](int power)
    {
        // Just manipulate exponent bits
        //static constexpr int IEEE754_EXPONENT_COUNT = 8;
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
        //return log2f(-power);
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
    // corner = 0 + (exp2f(morton3BitCount + 1) if(axis bit is 1 else 0)
    // for each dimension
    // if we pop up a level from (0.5, 0.5) and the bit are 11
    // p = 0.5 - (exp2f(morton3BitCount + 1))  if(axis bit is 1 else 0)
    uint64_t mortonBitStack = 0;
    uint8_t stack3BitCount = 0;

    // Helper Lambdas for readability (Basic stack functionality over the variables)
    auto PushMortonCode = [&mortonBitStack, &stack3BitCount](uint32_t bitCode)
    {
        assert(bitCode < 8);
        mortonBitStack = (mortonBitStack << 3) | bitCode;
        stack3BitCount += 1;

        //printf("PUSH %u -> %llX\n", static_cast<uint32_t>(stack3BitCount),
        //       mortonBitStack);

        assert(stack3BitCount <= 21);
    };
    auto PopMortonCode = [&mortonBitStack, &stack3BitCount]()
    {
        mortonBitStack >>= 3;
        stack3BitCount -= 1;
        //printf("POP %u -> %llX\n", static_cast<uint32_t>(stack3BitCount),
        //       mortonBitStack);
    };
    auto ReadMortonCode = [&mortonBitStack]() -> uint32_t
    {
        return mortonBitStack & 0b111;
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
    // (SVO is in [0,1]
    // Translate the ray to -AABBmin
    // Scale the ray from  [AABBmin, AABBmax] to [0,1]
    const Vector3f svoTranslate = -svoAABB.Min();
    const Vector3f svoScale = Vector3f(1.0f) / (svoAABB.Max() - svoAABB.Min());
    rDir = rDir * svoScale;
    rPos = (rPos + svoTranslate) * svoScale;
    // Now svo span calculations become trivial
    // 1.0f, 0.5f, 0.25f ...
    // You can even convert this to integer arithmetic (by changing exponent
    // only but it is out of scope of this code base)

    // Now also align all the rays as if their direction is positive
    // Mirror the SVO appropriately
    rPos[0] = (rDir[0] < 0.0f) ? (1.0f - rPos[0]) : rPos[0];
    rPos[1] = (rDir[1] < 0.0f) ? (1.0f - rPos[1]) : rPos[1];
    rPos[2] = (rDir[2] < 0.0f) ? (1.0f - rPos[2]) : rPos[2];
    // Mask will flip the childId to mirrored childId
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
    Vector3f posBias = rPos * dirCoeff;

    // Do initial AABB check with ray's tMin
    Vector3f tMaxXYZ = (dirCoeff - posBias) - Vector3f(tMin);
    Vector3f tMinXYZ = (         - posBias) - Vector3f(tMin);

    // Since every ray has positive direction, "tMin" will have the
    // minimum values.
    // Normally for every orientation (wrt. ray dir)
    // you need to check differently the "tMax" corner as well
    // here we don't need to
    tMinXYZ = Vector3f::Max(tMinXYZ, 0.0f);
    tMaxXYZ = Vector3f::Max(tMaxXYZ, 0.0f);
    float tMinSVO = min(min(tMinXYZ[0], tMinXYZ[1]), tMinXYZ[2]);
    float tMaxSVO = min(min(tMaxXYZ[0], tMaxXYZ[1]), tMaxXYZ[2]);

    // Check if ray is already out of bounds
    if(tMin > tMaxSVO) return tMaxSVO;
    // Ray is outside the svo but it can hit update tMin
    if(tMin < tMinSVO) tMin += tMinSVO;
    // Ray would never reach the svo return tMax as if we traversed to the fullest
    if(tMax < tMinSVO) return tMax;
    // Continuing case:
    // Ray is inside the SVO thus, tMin stays same

    // Traversal initial data
    // 0 index is root
    uint32_t nodeId = 0;
    // Initially corner is zero (SVO space [0,1])
    Vector3f corner = Zero3f;
    // Empty node boolean will be used to
    // create cleaner code
    // There should be at least one node
    // available on the tree (root)
    bool emptyNode = false;
    // Ray March Loop
    while(nodeId != UINT32_MAX)
    {
        // Children Voxel Span (derived from the level)
        float childVoxSpan = FastNegExp2f(stack3BitCount + 1);
        float currentVoxSpan = FastNegExp2f(stack3BitCount);
        // Find Center
        Vector3f voxCenter = corner + Vector3f(childVoxSpan);
        Vector3f voxTopRight = corner + Vector3f(currentVoxSpan);

        // Check early pop
        Vector3f tMaxXYZ = (voxTopRight * dirCoeff - posBias) - Vector3f(tMin);
        float tMax = min(min(tMaxXYZ[0], tMaxXYZ[1]), tMaxXYZ[2]);

        if(tMax <= 0.0f)
        {
            printf("[%u], FAILED tMax(%f) neg (Level %u)\n",
                   static_cast<uint32_t>(emptyNode), tMax,
                   static_cast<uint32_t>(stack3BitCount));
        }

        // We will "fake" traverse this empty node by just advancing
        // the tMin
        if(emptyNode)
        {
            // Find the smallest non-negative tValue
            // Advance the tMin accordingly
            tMin += tMax + MathConstants::Epsilon;
            // Nudge the tMin here
            //tMin = nextafterf(tMin, INFINITY);
            uint32_t mortonBits = ReadMortonCode();
            corner -= Vector3f(((mortonBits >> 0) & 0b1) ? childVoxSpan : 0.0f,
                               ((mortonBits >> 1) & 0b1) ? childVoxSpan : 0.0f,
                               ((mortonBits >> 2) & 0b1) ? childVoxSpan : 0.0f);
            //printf("[1] - POP: tMin %f  tMax %f NodeID %u Corner(%f, %f, %f)\n",
            //       tMin, tMax, nodeId,
            //       corner[0], corner[1], corner[2]);
            PopMortonCode();
            emptyNode = false;
            continue;
        }

        // We can fetch the node now
        uint64_t node = dNodes[nodeId];
        // Actual Traversal code
        // If out of bounds directly pop to parent
        if(tMax < 0.0f)
        {
            // Update the corner back according to the stack
            uint32_t mortonBits = ReadMortonCode();
            corner -= Vector3f(((mortonBits >> 0) & 0b1) ? childVoxSpan : 0.0f,
                               ((mortonBits >> 1) & 0b1) ? childVoxSpan : 0.0f,
                               ((mortonBits >> 2) & 0b1) ? childVoxSpan : 0.0f);
            PopMortonCode();
            // Actually pop to the parent
            //printf("[0] - POP: tMin %f tMax %f NodeID %u Corner(%f, %f, %f)\n",
            //       tMin, tMax, nodeId,
            //       corner[0], corner[1], corner[2]);
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
            Vector3f tMinXYZ = (voxCenter * dirCoeff - posBias);// -Vector3f(tMin);
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

                // If it is leaf just return the "nodeId" it is actually
                // leaf id
                if(IsChildrenLeaf(node))
                {
                    leafId = nodeId;
                    //printf("Found Leaf!\n");
                    break;
                }
            }
            // If this node does not have a child
            // "Fake" traverse down to this node (don't update the node)

            // Set empty node for next iteration
            emptyNode = !HasChild(node, mirChildId);
            // Unfortunately it is not leaf
            // Traverse down, update Corner
            corner += Vector3f(((childId >> 0) & 0b1) ? childVoxSpan : 0.0f,
                               ((childId >> 1) & 0b1) ? childVoxSpan : 0.0f,
                               ((childId >> 2) & 0b1) ? childVoxSpan : 0.0f);
            // Push the non-mirrored child id to the mask
            // (since corners are in mirrored coordinates)
            //printf("[%u] - PUSH: tMin %f NodeID %u Corner(%f, %f, %f)\n",
            //       static_cast<uint32_t>(emptyNode), tMin, nodeId,
            //       corner[0], corner[1], corner[2]);
            PushMortonCode(childId);
        }
    }
    // If code returns from here it means we could not find any child
    // All Done!
    return tMin;
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