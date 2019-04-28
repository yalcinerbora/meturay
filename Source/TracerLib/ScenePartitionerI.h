#pragma once

#include <vector>
#include <map>
#include <set>

struct SceneError;
struct SceneFileNode;
struct MatBatchData;
class CudaGPU;

using NodeListing = std::set<SceneFileNode>;

using MaterialNodeList = std::map<std::string, NodeListing>;
using MaterialBatchList = std::map<std::string, MatBatchData>;

using MultiGPUMatNodes = std::map<std::pair<std::string, int>, NodeListing>;
using MultiGPUMatBatches = std::map<std::pair<std::string, int>, MatBatchData>;

class ScenePartitionerI
{
	public:
		virtual					~ScenePartitionerI() = default;

		// Interface
		virtual SceneError		PartitionMaterials(MultiGPUMatNodes&,
												   MultiGPUMatBatches&,
												   int& boundaryMaterialGPU,
												   // Single Input
												   const MaterialNodeList& materialGroups,
												   const MaterialBatchList& materialBatches) const = 0;
};