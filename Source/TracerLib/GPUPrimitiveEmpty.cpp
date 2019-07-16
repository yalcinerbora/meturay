#include "GPUPrimitiveEmpty.h"

GPUPrimitiveEmpty::GPUPrimitiveEmpty()
{}

const char* GPUPrimitiveEmpty::Type() const
{
    return TypeName();
}

SceneError GPUPrimitiveEmpty::InitializeGroup(const NodeListing& surfaceDatalNodes, double time,
                                              const SurfaceLoaderGeneratorI&)
{
    return SceneError::OK;
}

SceneError GPUPrimitiveEmpty::ChangeTime(const NodeListing& surfaceDatalNodes, double time,
                                         const SurfaceLoaderGeneratorI&)
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

bool GPUPrimitiveEmpty::CanGenerateData(const std::string& s) const
{
    return false;
}