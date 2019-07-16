#include "BasicSurfaceLoaders.h"

#include "SceneIO.h"
#include "Sphere.h"
#include "Triangle.h"
#include "PrimitiveDataTypes.h"
#include "SceneNodeI.h"

#include <numeric>

InNodeTriLoader::InNodeTriLoader(const SceneNodeI& node, double time)
    : SurfaceLoader(node, time)
{
    if(node.IdCount() != 1) throw SceneException(SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR,
                                                 "InNodeTriangle cannot have multi id node.");
}

const char* InNodeTriLoader::SufaceDataFileExt() const
{
    return TypeName();
}

SceneError InNodeTriLoader::PrimDataLayout(PrimitiveDataLayout* result, PrimitiveDataType primitiveDataType) const
{
    if(primitiveDataType == PrimitiveDataType::POSITION ||
        primitiveDataType == PrimitiveDataType::NORMAL)
        result[0] = PrimitiveDataLayout::FLOAT_3;
    else if(primitiveDataType == PrimitiveDataType::UV)
        result[0] = PrimitiveDataLayout::FLOAT_1;
    else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
    return SceneError::OK;
}

SceneError InNodeTriLoader::BatchOffsets(size_t* result) const
{
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];

    size_t count = node.AccessListTotalCount(positionName);
    if(count % 3 != 0) 
        return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

    result[0] = 0;
    result[1] = count / 3;
    return SceneError::OK;
}

SceneError InNodeTriLoader::PrimitiveCounts(size_t* result) const
{
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];

    size_t count = node.AccessListTotalCount(positionName);
    if(count % 3 != 0) 
        return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;
    result[0] = (count / 3);
    return SceneError::OK;
}

SceneError InNodeTriLoader::AABB(AABB3* result) const
{
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    std::vector<Vector3> positions = node.AccessVector3List(positionName, time);

    if((positions.size() % 3) != 0)
        return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

    result[0] = AABB3(Zero3, Zero3);
    for(size_t i = 0; i < (positions.size() / 3); i++)
    {
        result[0].UnionSelf(Triangle::BoundingBox(positions[i * 3 + 0],
                                                  positions[i * 3 + 1],
                                                  positions[i * 3 + 2]));
    }
    return SceneError::OK;
}

SceneError InNodeTriLoader::GetPrimitiveData(Byte* memory, PrimitiveDataType primitiveDataType) const
{
    const int posIndex = static_cast<int>(primitiveDataType);
    const std::string name = PrimitiveDataTypeNames[posIndex];

    if(primitiveDataType == PrimitiveDataType::POSITION ||
       primitiveDataType == PrimitiveDataType::NORMAL)
    {
        std::vector<Vector3> data = node.AccessVector3List(name, time);
        std::copy(data.begin(), data.end(), reinterpret_cast<Vector3*>(memory));
        return SceneError::OK;
    }
    else if(primitiveDataType == PrimitiveDataType::UV)
    {
        std::vector<Vector2> data = node.AccessVector2List(name, time);
        std::copy(data.begin(), data.end(), reinterpret_cast<Vector2*>(memory));
        return SceneError::OK;
    }
    else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
}

InNodeSphrLoader::InNodeSphrLoader(const SceneNodeI& node, double time)
    : SurfaceLoader(node, time)
{}

// Type Determination
const char* InNodeSphrLoader::SufaceDataFileExt() const
{
    return TypeName();
}

SceneError InNodeSphrLoader::PrimDataLayout(PrimitiveDataLayout* result, PrimitiveDataType primitiveDataType) const
{
    for(size_t i = 0; i < node.IdCount(); i++)
    {
        if(primitiveDataType == PrimitiveDataType::POSITION)
            result[i] = PrimitiveDataLayout::FLOAT_3;
        else if(primitiveDataType == PrimitiveDataType::RADIUS)
            result[i] = PrimitiveDataLayout::FLOAT_1;
        else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
    }
    return SceneError::OK;
}

SceneError InNodeSphrLoader::BatchOffsets(size_t* result) const
{
    std::iota(result, result + (node.IdCount() + 1), 0);
    return SceneError::OK;
}

SceneError InNodeSphrLoader::PrimitiveCounts(size_t* result) const
{
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    size_t primDataCount = node.IdCount();

    std::fill_n(result, primDataCount, 1);
    return SceneError::OK;
}

SceneError InNodeSphrLoader::AABB(AABB3* result) const
{
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const int radIndex = static_cast<int>(PrimitiveDataType::RADIUS);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    const std::string radiusName = PrimitiveDataTypeNames[radIndex];

    std::vector<Vector3> positions = node.AccessVector3(positionName, time);
    std::vector<float> radiuses = node.AccessFloat(radiusName, time);

    if(positions.size() != node.IdCount() || radiuses.size() != node.IdCount())
        return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

    for(size_t i = 0; i < node.IdCount(); i++)
    {
        result[i] = Sphere::BoundingBox(positions[i], radiuses[i]);
    }
    return SceneError::OK;
}

SceneError InNodeSphrLoader::GetPrimitiveData(Byte* memory, PrimitiveDataType primitiveDataType) const
{
    int nameIndex = static_cast<int>(primitiveDataType);
    const std::string name = PrimitiveDataTypeNames[nameIndex];

    if(primitiveDataType == PrimitiveDataType::POSITION)
    {
        std::vector<Vector3> result = node.AccessVector3(name, time);
        std::copy(result.begin(), result.end(), reinterpret_cast<Vector3*>(memory));
        return SceneError::OK;
    }
    else if(primitiveDataType == PrimitiveDataType::RADIUS)
    {
        std::vector<float> result = node.AccessFloat(name, time);
        std::copy(result.begin(), result.end(), reinterpret_cast<float*>(memory));
        return SceneError::OK;
    }
    else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
}