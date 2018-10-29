#include "Materials.cuh"
#include "MaterialNodeReaders.h"

ColorMaterial::ColorMaterial()
{}

const char* ColorMaterial::Type() const
{
	return TypeName;
}
	
SceneError ColorMaterial::InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time)
{
	matData = ColorMaterialRead(memory, materialNodes, time);

	return SceneError::OK;
}

SceneError ColorMaterial::ChangeTime(const std::set<SceneFileNode>& materialNodes, double time)
{
	return SceneError::OK;
}

void ColorMaterial::LoadMaterial(uint32_t materialId, int gpuId)
{

}

void ColorMaterial::UnloadMaterial(uint32_t material)
{
}

int ColorMaterial::InnerId(uint32_t materialId) const
{
	return 0;
}

bool ColorMaterial::IsLoaded(uint32_t materialId) const
{
	return false;
}

size_t ColorMaterial::UsedGPUMemory() const
{
	return 0;
}

size_t ColorMaterial::UsedCPUMemory() const
{
	return 0;
}

size_t ColorMaterial::UsedGPUMemory(uint32_t materialId) const
{
	return 0;
}

size_t ColorMaterial::UsedCPUMemory(uint32_t materialId) const
{
	return 0;
}

uint8_t ColorMaterial::OutRayCount() const
{
	return 1;
}

// Material Batches
template class GPUMaterialBatch<TracerBasic,
	ColorMaterial,
	GPUPrimitiveTriangle,
	BasicSurfaceFromTri>;

template class GPUMaterialBatch<TracerBasic,
	ColorMaterial,
	GPUPrimitiveSphere,
	BasicSurfaceFromSphr>;