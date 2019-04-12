#include "MaterialNodeReaders.h"
#include "MaterialStructs.h"

#include "RayLib/SceneIO.h"

#include "TracerLib/DeviceMemory.h"
#include "TracerLib/SceneFileNode.h"

ConstantAlbedoMatData ConstantAlbedoMatRead(DeviceMemory& mem,
											const std::set<SceneFileNode>& materialNodes,
											double time)
{
	constexpr const char* ALBEDO = "albedo";

	std::vector<Vector3> albedoCPU;
	albedoCPU.reserve(materialNodes.size());

	for(const auto& sceneNode : materialNodes)
	{
		const nlohmann::json& node = sceneNode;
		albedoCPU.push_back(SceneIO::LoadVector<3, float>(node[ALBEDO], time));
	}

	// Alloc etc
	mem = std::move(DeviceMemory(albedoCPU.size() * sizeof(Vector3)));
	Vector3f* ptr = static_cast<Vector3f*>(mem);
	return {ptr};
}

ConstantBoundaryMatData ConstantBoundaryMatRead(const std::set<SceneFileNode>& materialNodes,
												double time)
{
	constexpr const char* ALBEDO = "albedo";
	if(materialNodes.size() == 0) return {};

	const SceneFileNode& sceneNode = *materialNodes.begin();
	const nlohmann::json& node = sceneNode;
	Vector3 background = SceneIO::LoadVector<3, float>(node[ALBEDO], time);
	return {background};
}