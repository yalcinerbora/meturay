#pragma once

#include "ScenePartitionerI.h"
#include "CudaConstants.h"

// Basic Partitioner (Throws everything to first GPU)
class SingleGPUScenePartitioner : public ScenePartitionerI
{
    private:
        const std::vector<CudaGPU>&     systemGPUs;

    protected:
    public:
        // Constructors & Destructor
                                        SingleGPUScenePartitioner(const std::vector<CudaGPU>&);
                                        ~SingleGPUScenePartitioner() = default;

        // Interface
        SceneError      PartitionMaterials(MultiGPUMatNodes&,
                                           MultiGPUMatBatches&,
                                           int&,
                                           // Single Input
                                           MaterialNodeList& materialGroups,
                                           MaterialBatchList& materialBatches) const override;
};