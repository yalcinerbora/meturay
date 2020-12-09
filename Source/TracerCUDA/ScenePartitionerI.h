#pragma once

#include <vector>
#include <map>
#include <set>
#include <string>

#include "NodeListing.h"

class CudaGPU;

struct SceneError;
struct WorkBatchData;

using MaterialNodeList = std::map<std::string, NodeListing>;
using WorkBatchList = std::map<std::string, WorkBatchData>;

using MultiGPUMatNodes = std::map<std::pair<std::string, const CudaGPU*>, NodeListing>;
using MultiGPUWorkBatches = std::map<std::pair<std::string, const CudaGPU*>, WorkBatchData>;

class ScenePartitionerI
{
    public:
        virtual                 ~ScenePartitionerI() = default;

        // Interface
        virtual SceneError      PartitionMaterials(MultiGPUMatNodes&,
                                                   MultiGPUWorkBatches&,
                                                   // Single Input
                                                   MaterialNodeList&,
                                                   WorkBatchList&) const = 0;
};