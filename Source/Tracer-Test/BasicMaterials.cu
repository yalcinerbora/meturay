#include "BasicMaterials.cuh"
#include "MaterialNodeReaders.h"

ConstantBoundaryMat::ConstantBoundaryMat(int gpuId)
    : GPUBoundaryMatGroup(gpuId)
{}

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

BasicMat::BasicMat(int gpuId)
    : GPUMaterialGroup(gpuId)
{}

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

BarycentricMat::BarycentricMat(int gpuId)
    : GPUMaterialGroup(gpuId)
{}

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

SphericalMat::SphericalMat(int gpuId)
    : GPUMaterialGroup(gpuId)
{}

SceneError SphericalMat::InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time)
{
    return SceneError::OK;
}

SceneError SphericalMat::ChangeTime(const std::set<SceneFileNode>& materialNodes, double time)
{
    return SceneError::OK;
}

int SphericalMat::InnerId(uint32_t materialId) const
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

template class GPUMaterialBatch<TracerBasic,
                                SphericalMat,
                                GPUPrimitiveSphere,
                                SphrSurfaceFromSphr>;