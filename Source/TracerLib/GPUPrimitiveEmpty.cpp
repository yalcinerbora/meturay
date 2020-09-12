#include "GPUPrimitiveEmpty.h"

GPUPrimitiveEmpty::GPUPrimitiveEmpty()
{}

const char* GPUPrimitiveEmpty::Type() const
{
    return TypeName();
}

SceneError GPUPrimitiveEmpty::InitializeGroup(const NodeListing& surfaceDatalNodes, double time,
                                              const SurfaceLoaderGeneratorI&,
                                              const std::string&)
{
    return SceneError::OK;
}

SceneError GPUPrimitiveEmpty::ChangeTime(const NodeListing& surfaceDatalNodes, double time,
                                         const SurfaceLoaderGeneratorI&,
                                         const std::string&)
{
    return SceneError::OK;
}

bool GPUPrimitiveEmpty::HasPrimitive(uint32_t surfaceDataId) const
{
    return false;
}

SceneError GPUPrimitiveEmpty::GenerateLights(std::vector<CPULight>&,
                                             const GPUDistribution2D&,
                                             HitKey key,
                                             uint32_t surfaceDataId,
                                             const Matrix4x4& transform) const
{
    return SceneError::OK;
}

Vector2ul GPUPrimitiveEmpty::PrimitiveBatchRange(uint32_t surfaceDataId) const
{
    return Zero2ul;
}

AABB3 GPUPrimitiveEmpty::PrimitiveBatchAABB(uint32_t surfaceDataId) const
{
    Vector3f minInf(-INFINITY);
    return AABB3f(minInf, minInf);
}

PrimTransformType GPUPrimitiveEmpty::TransformType() const
{
    return PrimTransformType::CONSTANT_LOCAL_TRANSFORM;
}

bool GPUPrimitiveEmpty::CanGenerateData(const std::string& s) const
{
    return false;
}