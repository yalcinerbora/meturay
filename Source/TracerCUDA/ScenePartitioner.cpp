#include "ScenePartitioner.h"

#include "RayLib/SceneStructs.h"
#include "RayLib/SceneError.h"

SingleGPUScenePartitioner::SingleGPUScenePartitioner(const CudaSystem& cudaSystem)
    : system(cudaSystem)
{}

// Algo assumes a single healthy GPU
SceneError SingleGPUScenePartitioner::PartitionMaterials(MultiGPUMatNodes& multiGroups,
                                                         MultiGPUWorkBatches& multiBatches,                                                     
                                                         // Single Input
                                                         MaterialNodeList& materialGroups,
                                                         WorkBatchList& workBatches) const
{
    // Just use the first gpu avail
    assert(!system.GPUList().empty());
    const CudaGPU& gpu = system.BestGPU();//*(system.GPUList().begin());

    for(auto& mg : materialGroups)
    {
        multiGroups.emplace(std::make_pair(mg.first, &gpu),
                            std::move(mg.second));
    }
    for(auto& mb : workBatches)
    {
        multiBatches.emplace(std::make_pair(mb.first, &gpu),
                             std::move(mb.second));
    }
    materialGroups.clear();
    workBatches.clear();
    return SceneError::OK;
}