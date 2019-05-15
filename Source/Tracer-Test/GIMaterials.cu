#include "GIMaterials.cuh"
#include "MaterialNodeReaders.h"

GIAlbedoMat::GIAlbedoMat(int gpuId)
    : GPUMaterialGroup(gpuId)
{}

SceneError GIAlbedoMat::InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time)
{
    dData = ConstantAlbedoMatRead(memory, materialNodes, time);
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

// Material Batch Implementations
template class GPUMaterialBatch<TracerBasic,
                                GIAlbedoMat,
                                GPUPrimitiveTriangle,
                                BasicSurfaceFromTri>;

template class GPUMaterialBatch<TracerBasic,
                                GIAlbedoMat,
                                GPUPrimitiveSphere,
                                BasicSurfaceFromSphr>;