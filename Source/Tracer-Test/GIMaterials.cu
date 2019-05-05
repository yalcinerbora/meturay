#include "GIMaterials.cuh"
#include "MaterialNodeReaders.h"

GIAlbedoMat::GIAlbedoMat(int gpuId)
    : GPUMaterialGroup(gpuId)
{}

const char* GIAlbedoMat::Type() const
{
    return TypeName();
}

SceneError GIAlbedoMat::InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time)
{
    matData = ConstantAlbedoMatRead(memory, materialNodes, time);
    return SceneError::OK;
}

SceneError GIAlbedoMat::ChangeTime(const std::set<SceneFileNode>& materialNodes, double time)
{
    return SceneError::OK;
}

int GIAlbedoMat::InnerId(uint32_t materialId) const
{
    return 0;
}

bool GIAlbedoMat::IsLoaded(uint32_t materialId) const
{
    return false;
}

size_t GIAlbedoMat::UsedGPUMemory() const
{
    return memory.Size();
}

size_t GIAlbedoMat::UsedCPUMemory() const
{
    return 0;
}

size_t GIAlbedoMat::UsedGPUMemory(uint32_t materialId) const
{
    return sizeof(Vector3f);
}

size_t GIAlbedoMat::UsedCPUMemory(uint32_t materialId) const
{
    return 0;
}

uint8_t GIAlbedoMat::OutRayCount() const
{
    return 1;
}

// Material Batch Implementations
template class GPUMaterialBatch<TracerBasic,
                                GIAlbedoMat,
                                GPUPrimitiveTriangle,
                                BasicSurfaceFromTri>;

template class GPUMaterialBatch<TracerBasic,
                                GIAlbedoMat,
                                GPUPrimitiveSphere,
                                BasicSurfaceFromSphr>;