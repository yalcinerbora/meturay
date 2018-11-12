#include "Materials.cuh"
#include "MaterialNodeReaders.h"

ConstantBoundaryMat::ConstantBoundaryMat()
{}

const char* ConstantBoundaryMat::Type() const
{
	return TypeName;
}

SceneError ConstantBoundaryMat::InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time)
{
	matData = ConstantBoundaryMatRead(materialNodes, time);
	return SceneError::OK;
}

SceneError ConstantBoundaryMat::ChangeTime(const std::set<SceneFileNode>& materialNodes, double time)
{

}

// Load/Unload Material			
void ConstantBoundaryMat::LoadMaterial(uint32_t materialId, int gpuId)
{}

void ConstantBoundaryMat::UnloadMaterial(uint32_t material)
{}

int ConstantBoundaryMat::InnerId(uint32_t materialId) const
{}

bool ConstantBoundaryMat::IsLoaded(uint32_t materialId) const
{}

size_t ConstantBoundaryMat::UsedGPUMemory() const
{}

size_t ConstantBoundaryMat::UsedCPUMemory() const
{}

size_t ConstantBoundaryMat::UsedGPUMemory(uint32_t materialId) const
{}

size_t ConstantBoundaryMat::UsedCPUMemory(uint32_t materialId) const
{}

uint8_t ConstantBoundaryMat::OutRayCount() const 
{
	return 0;
}

ConstantAlbedoMat::ConstantAlbedoMat()
{}

const char* ConstantAlbedoMat::Type() const
{
	return TypeName;
}
	
SceneError ConstantAlbedoMat::InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time)
{
	matData = ConstantAlbedoMatRead(memory, materialNodes, time);
	return SceneError::OK;
}

SceneError ConstantAlbedoMat::ChangeTime(const std::set<SceneFileNode>& materialNodes, double time)
{
	return SceneError::OK;
}

void ConstantAlbedoMat::LoadMaterial(uint32_t materialId, int gpuId)
{
	// Consider Textures here since no textures are avail ignore
}

void ConstantAlbedoMat::UnloadMaterial(uint32_t material)
{
	// Consider Textures here since no textures are avail ignore
}

int ConstantAlbedoMat::InnerId(uint32_t materialId) const
{
	return 0;
}

bool ConstantAlbedoMat::IsLoaded(uint32_t materialId) const
{
	return false;
}

size_t ConstantAlbedoMat::UsedGPUMemory() const
{
	return 0;
}

size_t ConstantAlbedoMat::UsedCPUMemory() const
{
	return 0;
}

size_t ConstantAlbedoMat::UsedGPUMemory(uint32_t materialId) const
{
	return 0;
}

size_t ConstantAlbedoMat::UsedCPUMemory(uint32_t materialId) const
{
	return 0;
}

uint8_t ConstantAlbedoMat::OutRayCount() const
{
	return 1;
}

// Material Batches
template class GPUBoundaryMatBatch<TracerBasic, ConstantAlbedoMatData>;

template class GPUMaterialBatch<TracerBasic,
	ConstantAlbedoMatData,
	GPUPrimitiveTriangle,
	BasicSurfaceFromTri>;

template class GPUMaterialBatch<TracerBasic,
	ConstantAlbedoMatData,
	GPUPrimitiveSphere,
	BasicSurfaceFromSphr>;