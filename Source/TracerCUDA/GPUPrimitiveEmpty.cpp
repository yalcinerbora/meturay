#include "GPUPrimitiveEmpty.h"

GPUPrimitiveEmpty::GPUPrimitiveEmpty()
{}

const char* GPUPrimitiveEmpty::Type() const
{
    return TypeName();
}

SceneError GPUPrimitiveEmpty::InitializeGroup(const NodeListing&, double,
                                              const SurfaceLoaderGeneratorI&,
                                              const TextureNodeMap&,
                                              const std::string&)
{
    return SceneError::OK;
}

SceneError GPUPrimitiveEmpty::ChangeTime(const NodeListing&, double,
                                         const SurfaceLoaderGeneratorI&,
                                         const std::string&)
{
    return SceneError::OK;
}

Vector2ul GPUPrimitiveEmpty::PrimitiveBatchRange(uint32_t) const
{
    return Zero2ul;
}

AABB3 GPUPrimitiveEmpty::PrimitiveBatchAABB(uint32_t) const
{
    Vector3f minInf(-INFINITY);
    return AABB3f(minInf, minInf);
}

bool GPUPrimitiveEmpty::PrimitiveBatchHasAlphaMap(uint32_t) const
{
    return false;
}

bool GPUPrimitiveEmpty::PrimitiveBatchBackFaceCulled(uint32_t surfaceDataId) const
{
    return true;
}

uint64_t GPUPrimitiveEmpty::TotalPrimitiveCount() const
{
    return 0;
}

uint64_t GPUPrimitiveEmpty::TotalDataCount() const
{
    return 0;
}

bool GPUPrimitiveEmpty::CanGenerateData(const std::string&) const
{
    return false;
}