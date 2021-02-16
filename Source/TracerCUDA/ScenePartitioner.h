#pragma once

#include "ScenePartitionerI.h"
#include "CudaConstants.h"

// Basic Partitioner (Throws everything to first GPU)
class SingleGPUScenePartitioner : public ScenePartitionerI
{
    private:
        const CudaSystem&   system;

    protected:
    public:
        // Constructors & Destructor
                            SingleGPUScenePartitioner(const CudaSystem&);
                            ~SingleGPUScenePartitioner() = default;

        // Interface        
        SceneError          PartitionMaterials(MultiGPUMatNodes&,
                                               MultiGPUWorkBatches&,
                                               // Single Input
                                               MaterialNodeList&,
                                               WorkBatchList&) const override;
};