#include "ScenePartitioner.h"

#include "RayLib/SceneStructs.h"
#include "RayLib/SceneError.h"

SingleGPUScenePartitioner::SingleGPUScenePartitioner(const CudaSystem& cudaSystem)
    : system(cudaSystem)
{}

// Algo assumes a single healthy GPU
SceneError SingleGPUScenePartitioner::PartitionMaterials(MultiGPUMatNodes& multiGroups,
                                                         MultiGPUMatBatches& multiBatches,
                                                         int& boundaryMaterialGPU,
                                                         // Single Input
                                                         MaterialNodeList& materialGroups,
                                                         MaterialBatchList& materialBatches) const
{
    // Just use the first gpu avail
    assert(!system.GPUList().empty());
    const CudaGPU& gpu = *(system.GPUList().begin());

    boundaryMaterialGPU = 0;

    for(auto& mg : materialGroups)
    {
        multiGroups.emplace(std::make_pair(mg.first, &gpu),
                            std::move(mg.second));
    }
    for(auto& mb : materialBatches)
    {
        multiBatches.emplace(std::make_pair(mb.first, &gpu),
                             std::move(mb.second));
    }
    materialGroups.clear();
    materialBatches.clear();
    return SceneError::OK;
}