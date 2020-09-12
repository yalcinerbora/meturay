#include "GPUMediumHomogenous.cuh"

SceneError CPUMediumHomogenous::InitializeGroup(const NodeListing& mediumNodes,
												double time,
												const std::string& scenePath)
{
	return SceneError::OK;
}

SceneError CPUMediumHomogenous::ChangeTime(const NodeListing& transformNodes, double time,
										   const std::string& scenePath)
{
	return SceneError::OK;
}

TracerError CPUMediumHomogenous::ConstructMediums(const CudaSystem&)
{
	return TracerError::OK;
}