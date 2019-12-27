#pragma once

#include <vector>
#include <map>
#include <set>
#include <string>

#include "NodeListing.h"

class CudaGPU;

struct SceneError;
struct MatBatchData;

using MaterialNodeList = std::map<std::string, NodeListing>;
using MaterialBatchList = std::map<std::string, MatBatchData>;

using MultiGPUMatNodes = std::map<std::pair<std::string, const CudaGPU*>, NodeListing>;
using MultiGPUMatBatches = std::map<std::pair<std::string, const CudaGPU*>, MatBatchData>;

class ScenePartitionerI
{
    public:
        virtual                 ~ScenePartitionerI() = default;

        // Interface
        virtual SceneError      PartitionMaterials(MultiGPUMatNodes&,
                                                   MultiGPUMatBatches&,
                                                   int& boundaryMaterialGPU,
                                                   // Single Input
                                                   MaterialNodeList& materialGroups,
                                                   MaterialBatchList& materialBatches) const = 0;
};