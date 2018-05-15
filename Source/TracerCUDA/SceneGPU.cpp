#include "SceneGPU.h"
#include "RayLib/RayHitStructs.h"

SceneGPU::SceneGPU()
{}

SceneGPU::SceneGPU(const std::string& fileName)
{}
	
const MaterialI& SceneGPU::Material(uint32_t) const
{

}

const MaterialList& SceneGPU::Materials() const
{

}

const SurfaceI& SceneGPU::Surface(uint32_t id) const
{

}

const SurfaceList& SceneGPU::Surfaces() const
{

}

AnimatableList& SceneGPU::Animatables()
{

}

const MeshBatchI& SceneGPU::MeshBatch(uint32_t id)
{

}

const MeshBatchList& SceneGPU::MeshBatch()
{

}

VolumeI& SceneGPU::Volume(uint32_t id)
{

}

VolumeList& SceneGPU::Volumes()
{

}

const LightI& SceneGPU::Light(uint32_t id)
{

}

void SceneGPU::ChangeTime(double timeSec)
{

}

void SceneGPU::HitRays(uint32_t* location,
					   const ConstRayRecordGMem,
					   uint32_t rayCount) const
{

}