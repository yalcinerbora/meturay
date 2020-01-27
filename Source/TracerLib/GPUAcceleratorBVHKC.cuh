#pragma once

#include "RayLib/HitStructs.h"
#include "RayLib/SceneStructs.h"
#include "RayLib/Types.h"

#include "AcceleratorDeviceFunctions.h"

enum class SplitAxis { X, Y, Z, END };

// Depth First Search over BVH
static constexpr const uint8_t MAX_DEPTH = 64;

static constexpr const uint8_t MAX_BASE_DEPTH = 27;
static constexpr const uint8_t MAX_BASE_DEPTH_BITS = 5;

static_assert((MAX_BASE_DEPTH + MAX_BASE_DEPTH_BITS) <= sizeof(uint32_t) * BYTE_BITS,
              "Base Accelerator State Bits should fit in 32-bit data.");
static_assert((1llu << MAX_BASE_DEPTH_BITS) >= MAX_BASE_DEPTH, 
              "Base Accelerator bits should hold depth.");

// Base BVH Kernel Save/Load State
__device__
inline static void LoadRayState(uint32_t& list, uint8_t& depth, uint32_t state)
{
    // MS side is list, LS side is depth
    list = (state >> MAX_BASE_DEPTH_BITS);
    depth = static_cast<uint8_t>(state & ((1u << MAX_BASE_DEPTH_BITS) - 1));
}

__device__
inline static uint32_t SaveRayState(uint32_t list, uint8_t depth)
{
    uint32_t state;
    state = (list << MAX_BASE_DEPTH_BITS);
    state &= (static_cast<uint32_t>(depth));
    return state;
}

struct SpacePartition
{
    private:
        float               splitPlane;
        SplitAxis           axis;
        const Vector3f*     dPrimCenters;

    protected:
    public:
        // Constructors & Destructor
        SpacePartition(float splitPlane, SplitAxis axis,
                       const Vector3f* dPCenters)
            : splitPlane(splitPlane)
            , axis(axis)
            , dPrimCenters(dPCenters)
        {}

        __device__ __host__
        __forceinline__ bool operator()(const uint32_t& id) const
        {
            int axisIndex = static_cast<int>(axis);
            // Get center location of tri
            float center = dPrimCenters[id][axisIndex];
            return center < splitPlane;
        }
};

template<class PGroup>
struct CentroidGen
{
    private:
        PGroup::PrimitiveData   pData;

    protected:
    public:
        // Constructors & Destructor                                
        CentroidGen(PGroup::PrimitiveData pData) : pData(pData) {}

        __device__ __host__
        __forceinline__ Vector3 operator()(const PrimitiveId& id) const
        {
            return PGroup::CenterFunc(id, pData);
        }
};

template<class PGroup>
struct AABBGen
{
    private:
        PGroup::PrimitiveData   pData;

    protected:
    public:
        // Constructors & Destructor                                
        AABBGen(PGroup::PrimitiveData pData) : pData(pData) {}

        __device__ __host__
        __forceinline__ AABB3f operator()(const PrimitiveId& id) const
        {
            return PGroup::BoxFunc(id, pData);
        }
};

struct AABBUnion
{
    __device__ __host__
    __forceinline__ AABB3f operator()(const AABB3f& a,
                                      const AABB3f& b) const
    {
        return a.Union(b);
    }
};

// Fundamental BVH Tree Node
template<class LeafStruct>
struct alignas(16) BVHNode
{
    //static constexpr uint32_t NULL_NODE = std::numeric_limits<uint32_t>::max();

    // Pointers
    union
    {
        // Non-leaf part
        struct
        {
            // 8 Word
            Vector3 aabbMin;
            uint32_t left;
            Vector3 aabbMax;
            uint32_t right;
        };
        // leaf part
        LeafStruct leaf;
    };
    uint32_t parent;
    bool isLeaf;
};

// Reductions
__global__
static void KCReduceAABBs(AABB3f* gAABBsReduced,
                          //
                          const AABB3f* gAABBs,
                          uint32_t count)
{
    static const AABB3 InitialAABB = NegativeAABB3;

    // Specialize BlockReduce for  256 Threads and flot
    typedef cub::BlockReduce<AABB3f, StaticThreadPerBlock1D> BlockReduce;
    // Shared Mem
    __shared__ typename BlockReduce::TempStorage tempStorage;

    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    AABB3f val = (globalId < count) ? gAABBs[globalId] : InitialAABB;

    AABB3f valReduced = BlockReduce(tempStorage).Reduce(val, AABBUnion());
    // Write
    if(threadIdx.x == 0) gAABBsReduced[blockIdx.x] = valReduced;
}

__global__
static void KCReduceAABBsFirst(AABB3f* gAABBsReduced,
                          //
                               const AABB3f* gAABBs,
                               const uint32_t* gIndices,
                               uint32_t count)
{
    static const AABB3 InitialAABB = NegativeAABB3;

    // Specialize BlockReduce for  256 Threads and flot
    typedef cub::BlockReduce<AABB3f, StaticThreadPerBlock1D> BlockReduce;
    // Shared Mem
    __shared__ typename BlockReduce::TempStorage tempStorage;

    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    AABB3f val = (globalId < count) ? gAABBs[gIndices[globalId]] : InitialAABB;

    AABB3f valReduced = BlockReduce(tempStorage).Reduce(val, AABBUnion());
    // Write
    if(threadIdx.x == 0) gAABBsReduced[blockIdx.x] = valReduced;
}

__global__
static void KCReduceCentroids(float* gCentersReduced,
                              //
                              const float* gCenters,
                              uint32_t count)
{
    // Specialize BlockReduce for  256 Threads and flot
    typedef cub::BlockReduce<float, StaticThreadPerBlock1D> BlockReduce;
     // Shared Mem
    __shared__ typename BlockReduce::TempStorage tempStorage;

    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (globalId < count) ? gCenters[globalId] : 0.0f;

    float valReduced = BlockReduce(tempStorage).Sum(val);

    // Write
    if(threadIdx.x == 0) gCentersReduced[blockIdx.x] = valReduced;
}

__global__
static void KCReduceCentroidsFirst(float* gCentersReduced,
                              //
                                   const Vector3f* gCenters,
                                   const uint32_t* gIndices,
                                   SplitAxis axis,
                                   uint32_t count)
{
    // Specialize BlockReduce for  256 Threads and flot
    typedef cub::BlockReduce<float, StaticThreadPerBlock1D> BlockReduce;
    // Shared Mem
    __shared__ typename BlockReduce::TempStorage tempStorage;

    int axisIndex = static_cast<int>(axis);

    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (globalId < count) ? gCenters[gIndices[globalId]][axisIndex] : 0.0f;

    float valReduced = BlockReduce(tempStorage).Sum(val);

    // Write
    if(threadIdx.x == 0) gCentersReduced[blockIdx.x] = valReduced;

}

__global__
static void KCInitIndices(// O
                          uint32_t* gIndices,
                          PrimitiveId* gPrimIds,
                          // Input
                          uint32_t indexStart,
                          uint64_t rangeStart,
                          uint32_t primCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        gIndices[globalId] = indexStart + globalId;
        gPrimIds[globalId] = rangeStart + globalId;
    }
}

template <class PGroup>
__global__
void KCGenCenters(// O
                  Vector3f* gCenters,
                  // Input
                  const uint32_t* gIndicies,
                  const PrimitiveId* gPrimitiveIds,
                  //
                  CentroidGen<PGroup> centFunc,
                  uint32_t primCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        uint32_t id = gIndicies[globalId];
        PrimitiveId primId = gPrimitiveIds[id];

        gCenters[globalId] = centFunc(primId);
    }
}

template <class PGroup>
__global__
void KCGenAABBs(// O
                AABB3f* gAABBs,
                // Input
                const uint32_t* gIndicies,
                const PrimitiveId* gPrimitiveIds,
                //
                AABBGen<PGroup> aabbFunc,
                uint32_t primCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        uint32_t id = gIndicies[globalId];
        PrimitiveId primId = gPrimitiveIds[id];

        gAABBs[globalId] = aabbFunc(primId);
    }
}

template <class PGroup>
__global__ __launch_bounds__(StaticThreadPerBlock1D)
void KCIntersectBVH(// O
                    HitKey* gMaterialKeys,
                    PrimitiveId* gPrimitiveIds,
                    HitStructPtr gHitStructs,
                    // I-O
                    RayGMem* gRays,
                    // Input
                    const TransformId* gTransformIds,
                    const RayId* gRayIds,
                    const HitKey* gAccelKeys,
                    const uint32_t rayCount,
                    // Constants
                    const BVHNode<PGroup::LeafData>** gBVHList,
                    const TransformStruct* gInverseTransforms,
                    //
                    const PGroup::PrimitiveData primData)
{
    using HitData = typename PGroup::HitData;       // HitRegister is defined by primitive
    using LeafData = typename PGroup::LeafData;     // LeafStruct is defined by primitive

    uint32_t sLocationStack[MAX_DEPTH];

    // Convenience Functions
    auto Push = [&sLocationStack](uint8_t& depth, uint32_t loc) -> void
    {
        //uint32_t index = StaticThreadPerBlock1D * depth + threadIdx.x;
        uint32_t index = depth;
        sLocationStack[index] = loc;
        depth++;
    };
    auto ReadTop = [&sLocationStack](uint8_t depth) -> uint32_t
    {
        //uint32_t index = StaticThreadPerBlock1D * depth + threadIdx.x;
        uint32_t index = depth;
        return sLocationStack[index];
    };
    auto Pop = [&ReadTop](uint8_t& depth) -> uint32_t
    {
        depth--;
        return ReadTop(depth);
    };

    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount; globalId += blockDim.x * gridDim.x)
    {
        const RayId id = gRayIds[globalId];
        const uint64_t accId = HitKey::FetchIdPortion(gAccelKeys[globalId]);
        const TransformId transformId = gTransformIds[id];

        // Load Ray/Hit to Register
        RayReg ray(gRays, id);

        // Key is the index of the inner BVH
        const BVHNode<LeafData>* gBVH = gBVHList[accId];

        // Zero means identity so skip
        if(transformId != 0)
        {
            TransformStruct s = gInverseTransforms[transformId];
            ray.ray.TransformSelf(s);
        }

        // Hit Data that is going to be fetched
        bool hitModified = false;
        HitKey materialKey;
        PrimitiveId primitiveId;
        HitData hit;

        uint8_t depth = 0;
        Push(depth, 0);
        const BVHNode<LeafData>* currentNode = nullptr;
        while(depth > 0)
        {
            uint32_t loc = Pop(depth);
            currentNode = gBVH + loc;

            //char debugChar = 'N';

            if(currentNode->isLeaf)
            {
                HitResult result = PGroup::HitFunc(// Output                                            
                                                   materialKey,
                                                   primitiveId,
                                                   hit,
                                                   // I-O
                                                   ray,
                                                   // Input
                                                   currentNode->leaf,
                                                   primData);

                hitModified |= result[1];
                if(result[0]) break;

            }
            else if(ray.ray.IntersectsAABB(currentNode->aabbMin, currentNode->aabbMax,
                                           Vector2f(ray.tMin, ray.tMax)))
            {
                // Push to stack
                Push(depth, currentNode->right);
                Push(depth, currentNode->left);
            }
        }
        // Write Updated Stuff
        if(hitModified)
        {
            ray.UpdateTMax(gRays, id);
            gHitStructs.Ref<HitData>(id) = hit;
            gMaterialKeys[id] = materialKey;
            gPrimitiveIds[id] = primitiveId;
        }
    }
}

// This is fundemental BVH traversal kernel
// It supparts partial traversal and continuation traversal(for scene tree)
template <class PGroup>
__global__ __launch_bounds__(StaticThreadPerBlock1D)
void KCIntersectBVHStackless(// O
                             HitKey* gMaterialKeys,
                             PrimitiveId* gPrimitiveIds,
                             HitStructPtr gHitStructs,
                             // I-O
                             RayGMem* gRays,
                             // Input
                             const TransformId* gTransformIds,
                             const RayId* gRayIds,
                             const HitKey* gAccelKeys,
                             const uint32_t rayCount,
                             // Constants
                             const BVHNode<PGroup::LeafData>** gBVHList,
                             const TransformStruct* gInverseTransforms,
                             //
                             const PGroup::PrimitiveData primData)
{
    using HitData = typename PGroup::HitData;       // HitRegister is defined by primitive
    using LeafData = typename PGroup::LeafData;     // LeafStruct is defined by primitive

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

    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount; globalId += blockDim.x * gridDim.x)
    {
        const RayId id = gRayIds[globalId];
        const uint64_t accId = HitKey::FetchIdPortion(gAccelKeys[globalId]);
        const TransformId transformId = gTransformIds[id];
        
        // Load Ray/Hit to Register
        RayReg ray(gRays, id);
        Vector2f tMinMax(ray.tMin, ray.tMax);
        
        // Key is the index of the inner BVH
        const BVHNode<LeafData>* gBVH = gBVHList[accId];

        // Zero means identity so skip
        if(transformId != 0)
        {
            TransformStruct s = gInverseTransforms[transformId];
            ray.ray.TransformSelf(s);
        }

        // Hit Data that is going to be fetched
        bool hitModified = false;
        HitKey materialKey;
        PrimitiveId primitiveId;
        HitData hit;

        uint64_t list = 0;
        uint8_t depth = MAX_DEPTH;
        const BVHNode<LeafData>* currentNode = gBVH;
        while(depth <= MAX_DEPTH)
        {
            // Determine traverse information
            uint8_t info = TraverseInfo(list, depth);
            // First time entry check intersection
            if(info == FIRST_ENTRY)                
            {
                // Leaf, so do its custom primitive intersection
                if(currentNode->isLeaf)
                {

                    HitResult result = PGroup::HitFunc(// Output                                            
                                                       materialKey,
                                                       primitiveId,
                                                       hit,
                                                       // I-O
                                                       ray,
                                                       // Input
                                                       currentNode->leaf,
                                                       primData);
                    hitModified |= result[1];
                    if(result[0]) break;

                    // Go Up
                    currentNode = gBVH + currentNode->parent;
                    Pop(list, depth);
                }
                // If not leaf do non-leaf intersection (AABB-Ray Intersection)
                else if(ray.ray.IntersectsAABB(currentNode->aabbMin, currentNode->aabbMax,
                                               Vector2f(ray.tMin, ray.tMax)))
                {
                    // Since by construction BVH tree has either no or both children
                    // avail. If a node is non-leaf it means that it has both of its children
                    // no need to check for left or right index validty
                    
                    // Directly go right
                    currentNode = gBVH + currentNode->left;
                    depth--;                

                }
                // If we could not be able to hit AABB
                // just go parent
                else
                {
                    currentNode = gBVH + currentNode->parent;
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
                currentNode = gBVH + currentNode->right;
                depth--;
            }
            // Now both left and right are processed
            // Just go up
            else
            {
                // Wipe out lower bits for incoming iterations
                WipeLowerBits(list, depth);
                currentNode = gBVH + currentNode->parent;
                depth++;
            }
        }
        // Write Updated Stuff 
        // if we did hit something
        if(hitModified)
        {
            ray.UpdateTMax(gRays, id);
            gHitStructs.Ref<HitData>(id) = hit;
            gMaterialKeys[id] = materialKey;
            gPrimitiveIds[id] = primitiveId;
        }
    }
}

__global__ __launch_bounds__(StaticThreadPerBlock1D)
static void KCIntersectBaseBVH(// Output
                               TransformId* gTransformIds,
                               HitKey* gHitKeys,
                               // I-O 
                               uint32_t* gRayStates,
                               uint32_t* gPrevBVHIndex,
                               // Input
                               const RayGMem* gRays,
                               const RayId* gRayIds,
                               const uint32_t rayCount,
                               // Constants
                               const BVHNode<BaseLeaf>* gBVH)
{    
    static constexpr uint8_t FIRST_ENTRY = 0b00;
    static constexpr uint8_t U_TURN = 0b01;

    // Convenience Functions
    auto WipeLowerBits = [](uint32_t& list, uint8_t depth)-> void
    {
        depth--;
        list &= (UINT64_MAX << depth);
    };
    auto TraverseInfo = [](uint32_t list, uint8_t depth) -> uint8_t
    {
        depth -= 2;
        return static_cast<uint8_t>((list >> depth) & 0x3ull);
    };
    auto MarkAsTraversed = [](uint32_t& list, uint8_t depth) -> void
    {
        depth--;
        list += (1u << depth);
    };
    auto Pop = [&MarkAsTraversed](uint32_t& list, uint8_t& depth) -> void
    {
        MarkAsTraversed(list, depth);
        depth++;
    };

    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount; globalId += blockDim.x * gridDim.x)
    {
        const RayId id = gRayIds[globalId];

        // Load Ray/Hit to Register
        RayReg rayData(gRays, id);
        Vector2f tMinMax(rayData.tMin, rayData.tMax);
        RayF& ray = rayData.ray;

        // Init Potential Hit Data
        HitKey nextAccKey = HitKey::InvalidKey;
        TransformId transformId = 0;
     
        // Initialize Previous States
        uint32_t list;
        uint8_t depth;
        LoadRayState(list, depth, gRayStates[id]);
                
        const BVHNode<BaseLeaf>* currentNode = gBVH + gPrevBVHIndex[id];
        while(depth <= MAX_BASE_DEPTH)
        {
            //if(id == 144)
            //    printf("[ ] [%03u %03u %03u] depth %u list 0x%016llX\n",
            //    (currentNode->isLeaf) ? 0 : currentNode->left,
            //           (currentNode->isLeaf) ? 0 : currentNode->right,
            //           currentNode->parent, depth, list);

            // Determine traverse loc
            uint8_t info = TraverseInfo(list, depth);
            // First time entry check
            if(info == FIRST_ENTRY)
            {
                bool isLeaf = currentNode->isLeaf;

                // Check the leafs aabb or non-leaf aabb
                // These are the same (cuz of union) but we need to be programatically correct
                // 
                Vector3 aabbMin = (isLeaf) ? currentNode->leaf.aabbMin : currentNode->aabbMin;
                Vector3 aabbMax = (isLeaf) ? currentNode->leaf.aabbMax : currentNode->aabbMax;

                // Check AABB
                if(ray.IntersectsAABB(aabbMin, aabbMax, tMinMax))
                {
                    // If we are at the leaf and found intersection
                    if(isLeaf)
                    {
                        // Pop to parent go get ready for next iteration
                        // However iteration will continue after Leaf Accelerators
                        Pop(list, depth);
                        // So save state                        
                        gRayStates[id] = SaveRayState(list, depth);
                        gPrevBVHIndex[id] = currentNode->parent;
                        // Set AcceleratorId and TransformId for lower accelerator
                        gHitKeys[globalId] = nextAccKey;
                        gTransformIds[id] = transformId;                                               
                        return;
                    }
                    // We are at non-leaf
                    else
                    {
                        // Go left on first entry
                        currentNode = gBVH + currentNode->left;
                        depth--;
                    }
               
                    //if(globalId == 144)
                    //    printf("[I]               depth %u list 0x%016llX\n", depth, list);

                }
                // No intersection
                // Turn back to parent
                else
                {                    
                    currentNode = gBVH + currentNode->parent;
                    Pop(list, depth);

                    //if(globalId == 144)
                    //    printf("[N]               depth %u list 0x%016llX\n", depth, list);
                }
            }
            // Doing U turn (left child to right child)
            // This means that we traversed left child and now
            // going to go to right child
            else if(info == U_TURN)
            {
                // Go to right child
                MarkAsTraversed(list, depth - 1);                
                currentNode = gBVH + currentNode->right;
                depth--;                
                
                //if(globalId == 144)
                //    printf("[U]               depth %u list 0x%016llX\n", depth, list);
            }
            // This means both left and right are processed
            // Go up
            else
            {
                // Wipe out lower bits for incoming iterations
                WipeLowerBits(list, depth);
                currentNode = gBVH + currentNode->parent;
                depth++;

                //if(globalId == 144)
                //    printf("[S]               depth %u list 0x%016llX\n", depth, list);
            }
            //if(globalId == 144) printf("----------\n");
        }

        // If all traverse is done 
        // write invalid key in order to terminate
        // rays accelerator traversal
        gHitKeys[globalId] = HitKey::InvalidKey;
    }
}