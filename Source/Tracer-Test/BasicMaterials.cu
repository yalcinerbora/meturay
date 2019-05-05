#include "BasicMaterials.cuh"
#include "MaterialNodeReaders.h"

ConstantBoundaryMat::ConstantBoundaryMat(int gpuId)
    : GPUBoundaryMatGroup(gpuId)
{}

const char* ConstantBoundaryMat::Type() const
{
    return TypeName();
}

SceneError ConstantBoundaryMat::InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time)
{
    dData = ConstantBoundaryMatRead(materialNodes, time);
    return SceneError::OK;
}

SceneError ConstantBoundaryMat::ChangeTime(const std::set<SceneFileNode>& materialNodes, double time)
{
    // TODO: Implement
    return SceneError::OK;
}

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

BasicMat::BasicMat(int gpuId)
    : GPUMaterialGroup(gpuId)
{}

const char* BasicMat::Type() const
{
    return TypeName();
}

SceneError BasicMat::InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time)
{
    dData = ConstantAlbedoMatRead(memory, materialNodes, time);
    return SceneError::OK;
}

SceneError BasicMat::ChangeTime(const std::set<SceneFileNode>& materialNodes, double time)
{
    return SceneError::OK;
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

BarycentricMat::BarycentricMat(int gpuId)
    : GPUMaterialGroup(gpuId)
{}

const char* BarycentricMat::Type() const
{
    return TypeName();
}

SceneError BarycentricMat::InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time)
{
    return SceneError::OK;
}

SceneError BarycentricMat::ChangeTime(const std::set<SceneFileNode>& materialNodes, double time)
{
    return SceneError::OK;
}

int BarycentricMat::InnerId(uint32_t materialId) const
{
    return 0;
}

bool BarycentricMat::IsLoaded(uint32_t materialId) const
{
    return true;
}

size_t BarycentricMat::UsedGPUMemory() const
{
    return 0;
}

size_t BarycentricMat::UsedCPUMemory() const
{
    return 0;
}

size_t BarycentricMat::UsedGPUMemory(uint32_t materialId) const
{
    return 0;
}

size_t BarycentricMat::UsedCPUMemory(uint32_t materialId) const
{
    return 0;
}

uint8_t BarycentricMat::OutRayCount() const
{
    return 0;
}

// Material Batches
template class GPUBoundaryMatBatch<TracerBasic, ConstantBoundaryMat>;

template class GPUMaterialBatch<TracerBasic,
                                BarycentricMat,
                                GPUPrimitiveTriangle,
                                BarySurfaceFromTri>;

template class GPUMaterialBatch<TracerBasic,
                                BasicMat,
                                GPUPrimitiveTriangle,
                                EmptySurfaceFromTri>;

template class GPUMaterialBatch<TracerBasic,
                                BasicMat,
                                GPUPrimitiveSphere,
                                EmptySurfaceFromSphr>;