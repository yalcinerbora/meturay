#include "KDTree.cuh"
#include <queue>


KDTreeCPU::KDTreeCPU()
{
    treeGPU.gLeafs = nullptr;
    treeGPU.gSplits = nullptr;
    treeGPU.gPackedData = nullptr;
    treeGPU.rootNodeId = UINT32_MAX;
}

TracerError KDTreeCPU::Construct(const Vector3f* dPositionList,
                                 uint32_t leafCount,
                                 const CudaSystem& system)
{
    static constexpr uint32_t MAX_BASE_DEPTH = 64;
    // Partition Function
    auto GenKdTreeNode = [](// Output
                            uint64_t& packedInfo,
                            float& splitPlane,
                            size_t& splitLoc,
                            bool& isLeaf,
                            // I-O
                            std::vector<Vector3f>& positions,
                            // Args
                            uint32_t childIndex,
                            uint32_t parentIndex,
                            size_t start, size_t end)
    {
        // Base Case
        if(end - start == 1)
        {
            splitLoc = std::numeric_limits<size_t>::max();
            KDTreeGPU::PackInfo(parentIndex,
                                start,
                                true,
                                KDTreeGPU::AXIS_END);
            isLeaf = true;
        }
        else
        {
            Vector3f maxPoint = Vector3f(-FLT_MAX);
            Vector3f minPoint = Vector3f(FLT_MAX);
            Vector3f center = Zero3f;
            for(size_t j = start; j < end; j++)
            {
                maxPoint = Vector3f::Max(maxPoint, positions[j]);
                minPoint = Vector3f::Min(minPoint, positions[j]);
                center += positions[j];
            }
            center /= (start - end);

            // Determine the split
            int maxIndex = (maxPoint - minPoint).Max();
            KDTreeGPU::AxisType axis = static_cast<KDTreeGPU::AxisType>(maxIndex);

            // Partition
            splitLoc = 0;
            int testAxis = maxIndex;
            // Partition wrt. avg center
            int64_t splitStart = static_cast<int64_t>(start - 1);
            int64_t splitEnd = static_cast<int64_t>(end);
            while(splitStart < splitEnd)
            {
                // Hoare Like Partition
                float leftAxisCenter;
                do
                {
                    if(splitStart >= static_cast<int64_t>(end - 1)) break;
                    splitStart++;
                    leftAxisCenter = positions[splitStart][testAxis];
                }
                while(leftAxisCenter >= center[testAxis]);
                float rightAxisCenter;
                do
                {
                    if(splitEnd <= static_cast<int64_t>(start + 1)) break;
                    splitEnd--;
                    rightAxisCenter = positions[splitEnd][testAxis];
                }
                while(rightAxisCenter <= center[testAxis]);

                if(splitStart < splitEnd)
                    std::swap(positions[splitEnd], positions[splitStart]);
            }
            // If cant find any proper split
            // Just cut in half
            if(splitLoc == 0) splitLoc = (end - start) / 2;

            // Sanity Check
            assert(splitLoc != start);
            assert(splitLoc != end);

            // Return
            splitPlane = center[static_cast<int>(axis)];
            KDTreeGPU::PackInfo(parentIndex,
                                childIndex,
                                false,
                                axis);
            isLeaf = false;
        }
    };

    // Load Leafs to Memory
    std::vector<Vector3f> hPositions(leafCount);
    CUDA_CHECK(cudaMemcpy(hPositions.data(), dPositionList,
                          sizeof(Vector3f) * leafCount,
                          cudaMemcpyDeviceToHost));
    // CPU Memory
    std::vector<uint64_t> hPackInfo;
    std::vector<float> hSplitPlanes;
    //
    struct SplitWork
    {
        bool isLeft;
        size_t start;
        size_t end;
        uint32_t parentId;
        uint32_t depth;
    };

    // Start Partitioning
    std::queue<SplitWork> partitionQueue;
    partitionQueue.emplace(SplitWork
                           {
                               false,
                               0, leafCount,
                               std::numeric_limits<uint32_t>::max(),
                               0
                           });

    // Breath first tree generation (top-down)
    uint8_t maxDepth = 0;
    while(!partitionQueue.empty())
    {
        SplitWork current = partitionQueue.front();
        partitionQueue.pop();

        size_t splitLoc;
        uint64_t packedInfo;
        float splitPlane;
        bool isLeaf;
        // Do Generation
        GenKdTreeNode(packedInfo,
                      splitPlane,
                      splitLoc,
                      isLeaf,
                      // I-O
                      hPositions,
                       // Args
                      static_cast<uint32_t>(hPackInfo.size() + 1),
                      current.parentId,
                      current.start, current.end);

        // Save
        hPackInfo.push_back(packedInfo);
        hSplitPlanes.push_back(splitPlane);

        // Next parent id
        uint32_t nextParentId = static_cast<uint32_t>(hPackInfo.size() - 1);

        // Update parent
        // Since nodes are adjacent only left can update the parent
        if(current.isLeft)
        {
            // Update the packed child Id of the parent
            KDTreeGPU::UpdateChildIndex(hPackInfo[current.parentId],
                                        nextParentId);
        }
        // Check if not base case and add more generation
        if(splitLoc != std::numeric_limits<size_t>::max())
        {
            partitionQueue.emplace(SplitWork{true, current.start, splitLoc, nextParentId, current.depth + 1});
            partitionQueue.emplace(SplitWork{false, splitLoc, current.end, nextParentId, current.depth + 1});
            maxDepth = static_cast<uint8_t>(current.depth + 1);

            if((current.depth + 1) > MAX_BASE_DEPTH)
                return TracerError::TRACER_INTERNAL_ERROR;
        }
    }
    // BVH cannot hold this surface return error
    if(maxDepth > MAX_BASE_DEPTH)
        return TracerError::TRACER_INTERNAL_ERROR;

    assert(hPackInfo.size() == hSplitPlanes.size());
    // Finally Allocate the entire node array
    uint64_t* dPackInfo;
    float* dSplitPlanes;
    Vector3f* dPositions;
    GPUMemFuncs::AllocateMultiData(std::tie(dPackInfo, dSplitPlanes, dPositions),
                                   memory,
                                   {hPackInfo.size(), hSplitPlanes.size(),
                                   hPositions.size()});
    CUDA_CHECK(cudaMemcpy(dPackInfo, hPackInfo.data(),
                          sizeof(uint64_t) * hPackInfo.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSplitPlanes, hSplitPlanes.data(),
                          sizeof(float) * hSplitPlanes.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPositions, dPositionList,
                          sizeof(Vector3f) * hPositions.size(),
                          cudaMemcpyDeviceToDevice));

    treeGPU.gPackedData = dPackInfo;
    treeGPU.gLeafs = dPositions;
    treeGPU.gSplits = dSplitPlanes;
    treeGPU.rootNodeId = 0;
    return TracerError::OK;
}


const KDTreeGPU& KDTreeCPU::TreeGPU() const
{
    return treeGPU;
}