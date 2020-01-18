#pragma once

#include "RayLib/HitStructs.h"
#include "RayLib/SceneStructs.h"

#include "AcceleratorDeviceFunctions.h"

enum class SplitAxis { X, Y, Z, END };

struct SpacePartition
{
    private:
        float                   splitPlane;
        SplitAxis               axis;
        const Vector3f*         dPrimCenters;
       
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
                                CentroidGen(PGroup::PrimitiveData pData)
            : pData(pData)
        {}

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
                                AABBGen(PGroup::PrimitiveData pData)
            : pData(pData)
        {}

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
struct alignas(8) BVHNode
{
    static constexpr uint32_t NULL_NODE = std::numeric_limits<uint32_t>::max();

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
            // 1 Word
            uint32_t parent;
        };
        // leaf part
        LeafStruct leaf;
    };
    bool isLeaf;
};

// Reductions
__global__
static void KCReduceAABBs(AABB3f* gAABBsReduced,
                          //
                          const AABB3f* gAABBs,
                          uint32_t count)
{
    // Specialize BlockReduce for  256 Threads and flot
    typedef cub::BlockReduce<AABB3f, StaticThreadPerBlock1D> BlockReduce;
    // Shared Mem
    __shared__ typename BlockReduce::TempStorage tempStorage;

    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    AABB3f val = (globalId < count) ? gAABBs[globalId]  : AABB3f(Zero3, Zero3);

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
    // Specialize BlockReduce for  256 Threads and flot
    typedef cub::BlockReduce<AABB3f, StaticThreadPerBlock1D> BlockReduce;
    // Shared Mem
    __shared__ typename BlockReduce::TempStorage tempStorage;
    
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    AABB3f val = (globalId < count) ? gAABBs[gIndices[globalId]] : AABB3f(Zero3, Zero3);

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
        uint32_t i = indexStart + globalId;
        gIndices[globalId] = i;
        gPrimIds[globalId] = rangeStart + i;
    }
}

template <class PGroup>
__global__ void KCGenCenters(// O
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
__global__ void KCGenAABBs(// O
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


// This is fundemental BVH traversal kernel
// It supparts partial traversal and continuation traversal(for scene tree)
template <class PGroup>
__global__ void KCIntersectBVH(// O
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

    // Convenience Functions
    auto IsAlreadyTraversed = [](uint64_t list, uint32_t depth) -> bool
    {
        return ((list >> depth) & 0x1) == 1;
    };
    auto MarkAsTraversed = [](uint64_t & list, uint32_t depth) -> void
    {
        list += (1 << depth);
    };
    auto Pop = [&MarkAsTraversed](uint64_t& list, uint32_t& depth) -> void
    {
        MarkAsTraversed(list, depth);
        depth++;
    };

    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount; globalId += blockDim.x * gridDim.x)
    {
        const RayId id = gRayIds[globalId];
        const uint64_t accId = HitKey::FetchIdPortion(gAccelKeys[id]);
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

        // Depth First Search over BVH
        uint32_t depth = sizeof(uint64_t) * 8;
        BVHNode<LeafData> currentNode = gBVH[0];
        for(uint64_t list = 0; list < UINT64_MAX;)
        {
            // Fast pop if both of the children is carries current node is zero
            // (This means that bit is carried)
            if(IsAlreadyTraversed(list, depth))
            {
                currentNode = gBVH[currentNode.parent];
                Pop(list, depth);
            }
            // Check if we already traversed left child
            // If child bit is on this means lower left child is traversed
            else if(IsAlreadyTraversed(list, depth - 1) &&
                    currentNode.right != BVHNode<LeafData>::NULL_NODE)
            {
                // Go to right child
                currentNode = gBVH[currentNode.right];
                depth--;
            }
            // Now this means that we entered to this node first time
            // Check if this node is leaf or internal
            // Check if it is leaf node
            else if(currentNode.isLeaf)
            {
                HitResult result = PGroup::HitFunc(// Output                                            
                                                   materialKey,
                                                   primitiveId,
                                                   hit,
                                                   // I-O
                                                   ray,
                                                   // Input
                                                   currentNode.leaf,
                                                   primData);
                hitModified |= result[1];
                if(result[0]) break;

                // Continue
                Pop(list, depth);
            }
            // Not leaf so check AABB
            else if(ray.ray.IntersectsAABB(currentNode.aabbMin, currentNode.aabbMax))
            {
                // Go left if avail
                if(currentNode.left != BVHNode<LeafData>::NULL_NODE)
                {
                    currentNode = gBVH[currentNode.left];
                    depth--;
                }
                // If not avail and since we are first time on this node
                // Try to go right
                else if(currentNode.right != BVHNode<LeafData>::NULL_NODE)
                {
                    // In this case dont forget to mark left child as traversed
                    MarkAsTraversed(list, depth - 1);

                    currentNode = gBVH[currentNode.right];
                    depth--;
                }
                else
                {
                    // This should not happen
                    // since we have "isNode" boolean
                    assert(false);

                    // Well in order to be correctly mark this node traversed also
                    // In the next iteration node will pop itself
                    MarkAsTraversed(list, depth - 1);
                }
            }
            // Finally no ray is intersected
            // Go to parent
            else
            {
                // Skip Leafs
                currentNode = gBVH[currentNode.parent];
                Pop(list, depth);
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