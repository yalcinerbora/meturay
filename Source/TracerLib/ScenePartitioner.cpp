#include "ScenePartitioner.h"

#include "RayLib/SceneStructs.h"
#include "RayLib/SceneError.h"
#include "SceneFileNode.h"

SingleGPUScenePartitioner::SingleGPUScenePartitioner(const std::vector<CudaGPU>& gList)
    : systemGPUs(gList)
{}

// Algo assumes a single healthy GPU
SceneError SingleGPUScenePartitioner::PartitionMaterials(MultiGPUMatNodes& multiGroups,
                                                         MultiGPUMatBatches& multiBatches,
                                                         int& boundaryMaterialGPU,
                                                         // Single Input
                                                         const MaterialNodeList& materialGroups,
                                                         const MaterialBatchList& materialBatches) const
{
    // Just use the first gpu avail
    assert(!systemGPUs.empty());
    const int GPUId = systemGPUs[0].DeviceId();

    boundaryMaterialGPU = 0;

    for(const auto& mg : materialGroups)
    {
        multiGroups.emplace(std::make_pair(mg.first, GPUId), mg.second);
    }
    for(const auto& mb : materialBatches)
    {
        multiBatches.emplace(std::make_pair(mb.first, GPUId), mb.second);
    }
    return SceneError::OK;
}