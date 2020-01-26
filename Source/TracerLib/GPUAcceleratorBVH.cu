#include "GPUAcceleratorBVH.cuh"
#include "TypeTraits.h"

#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"

const char* GPUBaseAcceleratorBVH::TypeName()
{
    return "BasicBVH";
}

GPUBaseAcceleratorBVH::GPUBaseAcceleratorBVH()
    : dBVH(nullptr)
    , dRayStates(nullptr)
{}

const char* GPUBaseAcceleratorBVH::Type() const
{
    return TypeName();
}

void GPUBaseAcceleratorBVH::GenerateBaseBVHNode(// Output
                                                size_t& splitLoc,
                                                BVHNode<BaseLeaf>& node,
                                                // Index Data                                             
                                                uint32_t* surfaceIndices,
                                                // Constants
                                                const BaseLeaf* leafs,
                                                const Vector3f* centers,
                                                // Call Related Args
                                                uint32_t parentIndex,
                                                SplitAxis axis,
                                                size_t start, size_t end)
{
    int axisIndex = static_cast<int>(axis);

    // Base Case (CPU Mode)
    if(end - start == 1)
    {
        uint32_t index = surfaceIndices[start];
        BaseLeaf leaf = leafs[index];

        node.isLeaf = true;
        node.leaf = leaf;
        splitLoc = std::numeric_limits<size_t>::max();
    }
    // Non-leaf Construct
    else
    {
        AABB3f aabbUnion = NegativeAABB3;
        float avgCenter = 0.0f;
        // Find AABB
        for(size_t i = start; i < end; i++)
        {
            uint32_t index = surfaceIndices[i];
            AABB3f aabb = AABB3f(leafs[index].aabbMin, leafs[index].aabbMax);
            float center = centers[index][axisIndex];
            aabbUnion.UnionSelf(aabb);
            avgCenter = (avgCenter * (i - start) + center) / (i - start + 1);
        }

        // Partition wrt. avg center
        int64_t splitStart = static_cast<int64_t>(start - 1);
        int64_t splitEnd = static_cast<int64_t>(end);
        while(splitStart < splitEnd)
        {
            // Hoare Like Partition
            float leftTriAxisCenter;
            do
            {
                if(splitStart >= static_cast<int64_t>(end - 1)) break;
                splitStart++;

                uint32_t index = surfaceIndices[splitStart];
                leftTriAxisCenter = centers[index][axisIndex];
            } while(leftTriAxisCenter >= avgCenter);
            float rightTriAxisCenter;
            do
            {
                if(splitEnd <= static_cast<int64_t>(start + 1)) break;
                splitEnd--;
                uint32_t index = surfaceIndices[splitEnd];
                rightTriAxisCenter = centers[index][axisIndex];
            } while(rightTriAxisCenter <= avgCenter);

            if(splitStart < splitEnd)
                std::swap(surfaceIndices[splitEnd], surfaceIndices[splitStart]);
        }
        node.aabbMin = aabbUnion.Min();
        node.aabbMax = aabbUnion.Max();
        assert(splitLoc != start);
        assert(splitLoc != end);
        splitLoc = splitStart;
    }
}

void GPUBaseAcceleratorBVH::GetReady(uint32_t rayCount)
{

}

void GPUBaseAcceleratorBVH::Hit(const CudaSystem&,
                                // Output
                                TransformId* dTransformIds,
                                HitKey* dAcceleratorKeys,
                                // Inputs
                                const RayGMem* dRays,
                                const RayId* dRayIds,
                                const uint32_t rayCount) const
{

}

SceneError GPUBaseAcceleratorBVH::Initialize(// List of surface to transform id hit key mappings
                                             const std::map<uint32_t, BaseLeaf>& map)
{
    idLookup.clear();
    leafs.resize(map.size());

    uint32_t i = 0;
    for(const auto& pair : map)
    {
        leafs[i] = pair.second;
        idLookup.emplace(pair.first, i);
        i++;
    }
    return SceneError::OK;
}

SceneError GPUBaseAcceleratorBVH::Change(// List of only changed surface to transform id hit key mappings
                                         const std::map<uint32_t, BaseLeaf>&)
{
    // TODO: Implement
    return SceneError::OK;
}

TracerError GPUBaseAcceleratorBVH::Constrcut(const CudaSystem&)
{
    // Alloc Id List & BVH Node List
    size_t totalSurfaceCount = idLookup.size();
    std::vector<uint32_t> surfaceIndices(totalSurfaceCount);    
    std::vector<BVHNode<BaseLeaf>> bvhNodes;

    // Generate Centers for Convenience
    std::vector<Vector3> surfaceCenters;
    for(const BaseLeaf& l : leafs)
    {
        Vector3 center = (l.aabbMin + l.aabbMax) * 0.5f;
        surfaceCenters.push_back(center);
    }

    // Gen recursive queue and do work
    struct SplitWork
    {
        bool left;
        size_t start;
        size_t end;
        SplitAxis axis;
        uint32_t parentId;
        uint32_t depth;
    };
    std::queue<SplitWork> partitionQueue;
    partitionQueue.emplace(SplitWork
                           {
                               false,
                               0, totalSurfaceCount,
                               SplitAxis::X,
                               std::numeric_limits<uint32_t>::max(),
                               0
                           });
    uint8_t maxDepth = 0;
    while(!partitionQueue.empty())
    {
        SplitWork current = partitionQueue.front();
        partitionQueue.pop();

        size_t splitLoc;
        BVHNode<BaseLeaf> node;

        // Do Generation
        GenerateBaseBVHNode(splitLoc,
                            node,
                            // Index Data
                            surfaceIndices.data(),
                            // Constants
                            leafs.data(),
                            surfaceCenters.data(),
                            // Call Related Args
                            current.parentId,
                            current.axis,
                            current.start, current.end);

        bvhNodes.emplace_back(node);
        uint32_t nextParentId = static_cast<uint32_t>(bvhNodes.size() - 1);
        SplitAxis nextSplit = DetermineNextSplit(current.axis, AABB3(node.aabbMin, node.aabbMax));

        // Update parent
        if(current.parentId != std::numeric_limits<uint32_t>::max())
        {
            if(current.left) bvhNodes[current.parentId].left = nextParentId;
            else bvhNodes[current.parentId].right = nextParentId;
        }

        // Check if not base case and add more generation
        if(splitLoc != std::numeric_limits<size_t>::max())
        {
            partitionQueue.emplace(SplitWork{true, current.start, splitLoc, nextSplit, nextParentId, current.depth + 1});
            partitionQueue.emplace(SplitWork{false, splitLoc, current.end, nextSplit, nextParentId, current.depth + 1});
            maxDepth = current.depth + 1;
        }
    }
     // BVH cannot hold this surface return error
    if(maxDepth <= MAX_BASE_DEPTH)
        return TracerError::UNABLE_TO_CONSTRUCT_BASE_ACCELERATOR;

    bvhMemory = DeviceMemory(bvhNodes.size() * sizeof(BVHNode<BaseLeaf>));
    dBVH = static_cast<const BVHNode<BaseLeaf>*>(bvhMemory);

    // Copy and All Done!
    CUDA_CHECK(cudaMemcpy(bvhMemory, bvhNodes.data(),
                          sizeof(BVHNode<BaseLeaf>)* bvhNodes.size(),
                          cudaMemcpyHostToDevice));

    return TracerError::OK;
}

TracerError GPUBaseAcceleratorBVH::Destruct(const CudaSystem&)
{
    return TracerError::OK;
}

// Accelerator Instancing for basic primitives
template class GPUAccBVHGroup<GPUPrimitiveTriangle>;
template class GPUAccBVHGroup<GPUPrimitiveSphere>;

template class GPUAccBVHBatch<GPUPrimitiveTriangle>;
template class GPUAccBVHBatch<GPUPrimitiveSphere>;

static_assert(IsTracerClass<GPUAccBVHGroup<GPUPrimitiveTriangle>>::value,
              "GPUAccBVHBatch<GPUPrimitiveTriangle> is not a Tracer Class.");
static_assert(IsTracerClass<GPUAccBVHGroup<GPUPrimitiveSphere>>::value,
              "GPUAccBVHGroup<GPUPrimitiveSphere> is not a Tracer Class.");