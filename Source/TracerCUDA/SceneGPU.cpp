#include "SceneGPU.h"
#include "RayLib/RayHitStructs.h"

SceneGPU::SceneGPU()
{}

SceneGPU::SceneGPU(const std::string& fileName)
{}
	
const MaterialI& SceneGPU::Material(uint32_t id) const
{
	return materials[id];
}

const MaterialList& SceneGPU::Materials() const
{
	return materials;
}

const SurfaceI& SceneGPU::Surface(uint32_t id) const
{
	return surfaces[id];
}

const SurfaceList& SceneGPU::Surfaces() const
{
	return surfaces;
}

AnimatableList& SceneGPU::Animatables()
{
	return animatables;
}

const MeshBatchI& SceneGPU::MeshBatch(uint32_t id)
{
	return  batches[id];
}

const MeshBatchList& SceneGPU::MeshBatch()
{
	return batches;
}

VolumeI& SceneGPU::Volume(uint32_t id)
{
	return volumes[id];
}

VolumeList& SceneGPU::Volumes()
{
	return volumes;
}

const LightI& SceneGPU::Light(uint32_t id)
{
	return lights[id];
}

void SceneGPU::ChangeTime(double timeSec)
{
}

void SceneGPU::HitRays(uint32_t* location,
					   const ConstRayRecordGMem,
					   uint32_t rayCount) const
{

}