#include "MaterialNodeReaders.h"
#include "MaterialStructs.h"

#include "RayLib/SceneFileNode.h"
#include "RayLib/SceneIO.h"

#include "TracerLib/DeviceMemory.h"

ColorMaterialData ColorMaterialRead(DeviceMemory& mem,
									const std::set<SceneFileNode>& materialNodes,
									double time)
{
	constexpr const char* ALBEDO = "albedo";

	std::vector<Vector3> albedoCPU;
	albedoCPU.reserve(materialNodes.size());

	for(const auto& node : materialNodes)
	{
		albedoCPU.push_back(SceneIO::LoadVector<3, float>(node.jsn[ALBEDO], time));
	}

	// Alloc etc
	mem = std::move(DeviceMemory(albedoCPU.size() * sizeof(Vector3)));
	Vector3f* ptr = static_cast<Vector3f*>(mem);
	return {ptr};
}