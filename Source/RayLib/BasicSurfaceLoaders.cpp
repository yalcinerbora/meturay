#include "BasicSurfaceLoaders.h"

#include "SceneIO.h"
#include "Sphere.h"
#include "Triangle.h"
#include "PrimitiveDataTypes.h"
#include "SceneNodeI.h"

#include <numeric>

InNodeTriLoader::InNodeTriLoader(const SceneNodeI& node, double time)
    : SurfaceLoader(node, time)
{}

const char* InNodeTriLoader::SufaceDataFileExt() const
{
    return TypeName();
}

SceneError InNodeTriLoader::PrimDataLayout(PrimitiveDataLayout* result, PrimitiveDataType primitiveDataType) const
{
    for(size_t i = 0; i < node.IdCount(); i++)
    {
        if(primitiveDataType == PrimitiveDataType::POSITION ||
           primitiveDataType == PrimitiveDataType::NORMAL)
            result[i] = PrimitiveDataLayout::FLOAT_3;
        else if(primitiveDataType == PrimitiveDataType::UV)
            result[i] = PrimitiveDataLayout::FLOAT_1;
        else if(primitiveDataType == PrimitiveDataType::VERTEX_INDEX)
            result[i] = PrimitiveDataLayout::FLOAT_1;
        else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
    }
    return SceneError::OK;
}

SceneError InNodeTriLoader::BatchOffsets(size_t* result) const
{
    size_t offset = 0;

    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    auto counts = node.AccessListCount(positionName);

    size_t i;
    for(i = 0; i < node.IdCount(); i++)
    {
        size_t primCount = counts[i];
        if(primCount % 3 != 0)
            return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

        result[i] = offset;
        offset += (primCount / 3);        
    }
    result[i] = offset;
    return SceneError::OK;
}

SceneError InNodeTriLoader::PrimitiveCounts(size_t* result) const
{
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    auto counts = node.AccessListCount(positionName);

    for(size_t i = 0; i < node.IdCount(); i++)
    {
        size_t primCount = counts[i];
        if(primCount % 3 != 0)
            return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;
        result[i] = (primCount / 3);
    }
    return SceneError::OK;
}

SceneError InNodeTriLoader::AABB(AABB3* result) const
{
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    std::vector<Vector3List> positions = node.AccessVector3List(positionName, time);

    for(size_t i = 0; i < node.IdCount(); i++)
    {
        const Vector3List posList = positions[i];
        if((posList.size() % 3) != 0)
            return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

        result[i] = AABB3(Zero3, Zero3);
        for(size_t i = 0; i < (posList.size() / 3); i++)
        {
            result[0].UnionSelf(Triangle::BoundingBox(posList[i * 3 + 0],
                                                      posList[i * 3 + 1],
                                                      posList[i * 3 + 2]));
        }
    }
    return SceneError::OK;
}

SceneError InNodeTriLoader::GetPrimitiveData(Byte* memory, PrimitiveDataType primitiveDataType) const
{
    const int index = static_cast<int>(primitiveDataType);
    const std::string name = PrimitiveDataTypeNames[index];

    if(primitiveDataType == PrimitiveDataType::POSITION ||
        primitiveDataType == PrimitiveDataType::NORMAL)
    {
        std::vector<Vector3List> data = node.AccessVector3List(name, time);

        size_t offset = 0;
        for(size_t i = 0; i < node.IdCount(); i++)
        {
            const Vector3List& currentList = data[i];
            std::copy(currentList.begin(), currentList.end(), reinterpret_cast<Vector3*>(memory + offset));
            offset += currentList.size();
        }
        return SceneError::OK;
    }
    else if(primitiveDataType == PrimitiveDataType::UV)
    {
        std::vector<Vector2List> data = node.AccessVector2List(name, time);
        size_t offset = 0;
        for(size_t i = 0; i < node.IdCount(); i++)
        {
            const Vector2List& currentList = data[i];
            std::copy(currentList.begin(), currentList.end(), reinterpret_cast<Vector2*>(memory + offset));
            offset += currentList.size();
            
        }
        return SceneError::OK;
    }
    else if(primitiveDataType == PrimitiveDataType::VERTEX_INDEX)
    {
        // If requested generate a dummy index list
        const int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
        const std::string name = PrimitiveDataTypeNames[posIndex];
        std::vector<size_t> counts = node.AccessListCount(name);
        size_t offset = 0;
        for(size_t i = 0; i < node.IdCount(); i++)
        {
            uint32_t* currentList = reinterpret_cast<uint32_t*>(memory + offset);
            std::iota(currentList, currentList + counts[i], 0);
            offset += counts[i];
        }
        return SceneError::OK;
    }
    else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
}

SceneError InNodeTriLoader::PrimitiveDataCount(size_t* result, PrimitiveDataType primitiveDataType) const
{
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    auto counts = node.AccessListCount(positionName);

    std::copy(counts.begin(), counts.end(), result);    
    return SceneError::OK;
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

SceneError InNodeSphrLoader::PrimitiveDataCount(size_t* result, PrimitiveDataType primitiveDataType) const
{
    return PrimitiveCounts(result);
}