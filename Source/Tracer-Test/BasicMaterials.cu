#include "BasicMaterials.cuh"
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
	// TODO: Implement
	return SceneError::OK;
}

// Load/Unload Material			
void ConstantBoundaryMat::LoadMaterial(uint32_t materialId, int gpuId)
{}

void ConstantBoundaryMat::UnloadMaterial(uint32_t material)
{}

int ConstantBoundaryMat::InnerId(uint32_t materialId) const
{
	return 0;
}

bool ConstantBoundaryMat::IsLoaded(uint32_t materialId) const
{
	return true;
}

size_t ConstantBoundaryMat::UsedGPUMemory() const
{
	return 0;
}

size_t ConstantBoundaryMat::UsedCPUMemory() const
{
	return sizeof(Vector3);
}

size_t ConstantBoundaryMat::UsedGPUMemory(uint32_t materialId) const
{
	return UsedGPUMemory();
}

size_t ConstantBoundaryMat::UsedCPUMemory(uint32_t materialId) const
{
	return UsedCPUMemory();
}

uint8_t ConstantBoundaryMat::OutRayCount() const 
{
	return 0;
}

BasicMat::BasicMat()
{}

const char* BasicMat::Type() const
{
	return TypeName;
}
	
SceneError BasicMat::InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time)
{
	matData = ConstantAlbedoMatRead(memory, materialNodes, time);
	return SceneError::OK;
}

SceneError BasicMat::ChangeTime(const std::set<SceneFileNode>& materialNodes, double time)
{
	return SceneError::OK;
}

void BasicMat::LoadMaterial(uint32_t materialId, int gpuId)
{
	// Consider Textures here since no textures are avail ignore
}

void BasicMat::UnloadMaterial(uint32_t material)
{
	// Consider Textures here since no textures are avail ignore
}

int BasicMat::InnerId(uint32_t materialId) const
{
	return 0;
}

bool BasicMat::IsLoaded(uint32_t materialId) const
{
	return false;
}

size_t BasicMat::UsedGPUMemory() const
{
	return memory.Size();
}

size_t BasicMat::UsedCPUMemory() const
{
	return 0;
}

size_t BasicMat::UsedGPUMemory(uint32_t materialId) const
{
	return sizeof(Vector3f);
}

size_t BasicMat::UsedCPUMemory(uint32_t materialId) const
{
	return 0;
}

uint8_t BasicMat::OutRayCount() const
{
	return 0;
}

// Material Batches
template class GPUBoundaryMatBatch<TracerBasic, ConstantBoundaryMat>;

template class GPUMaterialBatch<TracerBasic,
							    BasicMat,
							    GPUPrimitiveTriangle,
							    EmptySurfaceFromTri>;

template class GPUMaterialBatch<TracerBasic,
							    BasicMat,
							    GPUPrimitiveSphere,
							    EmptySurfaceFromSphr>;