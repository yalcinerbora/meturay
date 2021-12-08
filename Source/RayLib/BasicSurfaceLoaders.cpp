#include "BasicSurfaceLoaders.h"

#include "SceneIO.h"
#include "Sphere.h"
#include "Triangle.h"
#include "PrimitiveDataTypes.h"
#include "SceneNodeI.h"

#include <numeric>

//==================//
//  TRIANGLE LOADER //
//==================//
InNodeTriLoader::InNodeTriLoader(const SceneNodeI& node, double time)
    : SurfaceLoader(node, time)
{}

const char* InNodeTriLoader::SufaceDataFileExt() const
{
    return TypeName();
}

SceneError InNodeTriLoader::AABB(std::vector<AABB3>& result) const
{
    result.resize(node.IdCount());
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    std::vector<Vector3List> positions = node.AccessVector3List(positionName, time);

    for(size_t j = 0; j < node.IdCount(); j++)
    {
        const Vector3List posList = positions[j];
        if((posList.size() % 3) != 0)
            return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

        //result[j] = ZeroAABB3;
        result[j] = NegativeAABB3;
        for(size_t i = 0; i < (posList.size() / 3); i++)
        {
            result[j].UnionSelf(Triangle::BoundingBox(posList[i * 3 + 0],
                                                      posList[i * 3 + 1],
                                                      posList[i * 3 + 2]));
        }
    }
    return SceneError::OK;
}

SceneError InNodeTriLoader::PrimitiveRanges(std::vector<Vector2ul>& result) const
{
    result.resize(node.IdCount());
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    auto counts = node.AccessListCount(positionName);

    size_t offset = 0;
    for(size_t i = 0; i < node.IdCount(); i++)
    {
        size_t primCount = counts[i];
        if(primCount % 3 != 0)
            return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

        uint64_t begin = offset;
        offset += (primCount / 3);
        uint64_t end = offset;

        result[i] = Vector2ul(begin, end);
    }
    return SceneError::OK;
}

SceneError InNodeTriLoader::PrimitiveCounts(std::vector<size_t>& result) const
{
    result.resize(node.IdCount());
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    auto counts = node.AccessListCount(positionName);

    for(size_t i = 0; i < node.IdCount(); i++)
    {
        size_t primCount = counts[i];
        if(primCount % 3 != 0) return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;
        result[i] = (primCount / 3);
    }
    return SceneError::OK;
}

SceneError InNodeTriLoader::PrimitiveDataRanges(std::vector<Vector2ul>& result) const
{
    result.resize(node.IdCount());
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    auto counts = node.AccessListCount(positionName);

    size_t offset = 0;
    for(size_t i = 0; i < node.IdCount(); i++)
    {
        size_t primCount = counts[i];
        uint64_t begin = offset;
        offset += primCount;
        uint64_t end = offset;

        result[i] = Vector2ul(begin, end);
    }
    return SceneError::OK;
}

SceneError InNodeTriLoader::GetPrimitiveData(Byte* result, PrimitiveDataType primitiveDataType) const
{
    size_t offset = 0;
    PrimitiveDataType readPT = (primitiveDataType == PrimitiveDataType::VERTEX_INDEX) ?
                                    PrimitiveDataType::POSITION : primitiveDataType;
    const int index = static_cast<int>(readPT);
    const std::string name = PrimitiveDataTypeNames[index];

    if(primitiveDataType == PrimitiveDataType::POSITION ||
       primitiveDataType == PrimitiveDataType::NORMAL)
    {
        std::vector<Vector3List> data = node.AccessVector3List(name, time);
        for(size_t i = 0; i < node.IdCount(); i++)
        {
            const Vector3List& currentList = data[i];
            Vector3* writeLoc = reinterpret_cast<Vector3*>(result) + offset;

            std::copy(currentList.begin(), currentList.end(), writeLoc);
            offset += currentList.size();
        }
        return SceneError::OK;
    }
    else if(primitiveDataType == PrimitiveDataType::UV)
    {
        std::vector<Vector2List> data = node.AccessVector2List(name, time);
        for(size_t i = 0; i < node.IdCount(); i++)
        {
            const Vector2List& currentList = data[i];
            Vector2* writeLoc = reinterpret_cast<Vector2*>(result) + offset;
            std::copy(currentList.begin(), currentList.end(), writeLoc);
            offset += currentList.size();
        }
        return SceneError::OK;
    }
    else if(primitiveDataType == PrimitiveDataType::VERTEX_INDEX)
    {
        std::vector<size_t> counts = node.AccessListCount(name);

        for(size_t i = 0; i < node.IdCount(); i++)
        {
            uint64_t* writeLoc = reinterpret_cast<uint64_t*>(result) + offset;
            std::iota(writeLoc, writeLoc + counts[i], 0ull);
            offset += counts[i];
        }
        return SceneError::OK;
    }
    else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
}

SceneError InNodeTriLoader::HasPrimitiveData(bool& r, PrimitiveDataType primitiveDataType) const
{
    r = (primitiveDataType == PrimitiveDataType::POSITION ||
         primitiveDataType == PrimitiveDataType::NORMAL ||
         primitiveDataType == PrimitiveDataType::UV);
    return SceneError::OK;
}

SceneError InNodeTriLoader::PrimitiveDataCount(size_t& result, PrimitiveDataType primitiveDataType) const
{
    if(primitiveDataType == PrimitiveDataType::VERTEX_INDEX)
        primitiveDataType = PrimitiveDataType::POSITION;
    int posIndex = static_cast<int>(primitiveDataType);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    result = node.AccessListTotalCount(positionName);
    return SceneError::OK;
}

SceneError InNodeTriLoader::PrimDataLayout(PrimitiveDataLayout& result, PrimitiveDataType primitiveDataType) const
{
    if(primitiveDataType == PrimitiveDataType::POSITION ||
        primitiveDataType == PrimitiveDataType::NORMAL)
        result = PrimitiveDataLayout::FLOAT_3;
    else if(primitiveDataType == PrimitiveDataType::UV)
        result = PrimitiveDataLayout::FLOAT_2;
    else if(primitiveDataType == PrimitiveDataType::VERTEX_INDEX)
        result = PrimitiveDataLayout::UINT64_1;
    else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
    return SceneError::OK;
}

//=========================//
// TRIANGLE INDEXED LOADER //
//=========================//
InNodeTriLoaderIndexed::InNodeTriLoaderIndexed(const SceneNodeI& node, double time)
    : SurfaceLoader(node, time)
{}

// Type Determination
const char* InNodeTriLoaderIndexed::SufaceDataFileExt() const
{
    return TypeName();
}

// Per Batch Fetch
SceneError InNodeTriLoaderIndexed::AABB(std::vector<AABB3>& result) const
{
    result.resize(node.IdCount());
    int vertIndex = static_cast<int>(PrimitiveDataType::VERTEX_INDEX);
    const std::string indexName = PrimitiveDataTypeNames[vertIndex];
    auto indices = node.AccessUInt64List(indexName);

    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    auto positions = node.CommonVector3List(positionName);

    for(size_t j = 0; j < node.IdCount(); j++)
    {
        const UInt64List& indexList = indices[j];
        if((indexList.size() % 3) != 0)
            return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

        //result[j] = ZeroAABB3;
        result[j] = NegativeAABB3;
        for(size_t i = 0; i < (indexList.size() / 3); i++)
        {
            result[j].UnionSelf(Triangle::BoundingBox(positions[indexList[i * 3 + 0]],
                                                      positions[indexList[i * 3 + 1]],
                                                      positions[indexList[i * 3 + 2]]));
        }
    }
    return SceneError::OK;
}

SceneError InNodeTriLoaderIndexed::PrimitiveRanges(std::vector<Vector2ul>& result) const
{
    result.resize(node.IdCount());
    int posIndex = static_cast<int>(PrimitiveDataType::VERTEX_INDEX);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    auto counts = node.AccessListCount(positionName);

    size_t offset = 0;
    for(size_t i = 0; i < node.IdCount(); i++)
    {
        size_t primCount = counts[i];
        if(primCount % 3 != 0)
            return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

        uint64_t begin = offset;
        offset += (primCount / 3);
        uint64_t end = offset;

        result[i] = Vector2ul(begin, end);
    }
    return SceneError::OK;
}

SceneError InNodeTriLoaderIndexed::PrimitiveCounts(std::vector<size_t>& result) const
{
    result.resize(node.IdCount());
    int posIndex = static_cast<int>(PrimitiveDataType::VERTEX_INDEX);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    auto counts = node.AccessListCount(positionName);

    std::transform(counts.begin(), counts.end(), result.begin(),
                   [](uint64_t data) -> size_t { return data / 3; });
    return SceneError::OK;
}

SceneError InNodeTriLoaderIndexed::PrimitiveDataRanges(std::vector<Vector2ul>& result) const
{
    // We assume data is shared between all primitives
    // So data range is [0-positionCount]
    const int index = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string name = PrimitiveDataTypeNames[index];
    size_t size = node.CommonListSize(name);

    result.resize(node.IdCount());
    std::fill(result.begin(), result.end(), Vector2ul(0, size));
    return SceneError::OK;
}

SceneError InNodeTriLoaderIndexed::GetPrimitiveData(Byte* result, PrimitiveDataType primitiveDataType) const
{
    const int index = static_cast<int>(primitiveDataType);
    const std::string name = PrimitiveDataTypeNames[index];

    if(primitiveDataType == PrimitiveDataType::POSITION ||
       primitiveDataType == PrimitiveDataType::NORMAL)
    {
        Vector3List data = node.CommonVector3List(name, time);
        Vector3* writeLoc = reinterpret_cast<Vector3*>(result);
        std::copy(data.begin(), data.end(), writeLoc);
        return SceneError::OK;
    }
    else if(primitiveDataType == PrimitiveDataType::UV)
    {
        Vector2List data = node.CommonVector2List(name, time);
        Vector2* writeLoc = reinterpret_cast<Vector2*>(result);
        std::copy(data.begin(), data.end(), writeLoc);
        return SceneError::OK;
    }
    else if(primitiveDataType == PrimitiveDataType::VERTEX_INDEX)
    {
        size_t offset = 0;
        std::vector<UInt64List> data = node.AccessUInt64List(name, time);
        for(size_t i = 0; i < node.IdCount(); i++)
        {
            const UInt64List& currentList = data[i];
            uint64_t* writeLoc = reinterpret_cast<uint64_t*>(result) + offset;
            std::copy(currentList.begin(), currentList.end(), writeLoc);
            offset += currentList.size();
        }
        return SceneError::OK;
    }
    else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
}

SceneError InNodeTriLoaderIndexed::HasPrimitiveData(bool& r, PrimitiveDataType primitiveDataType) const
{
    r = (primitiveDataType == PrimitiveDataType::POSITION ||
         primitiveDataType == PrimitiveDataType::NORMAL ||
         primitiveDataType == PrimitiveDataType::UV ||
         primitiveDataType == PrimitiveDataType::VERTEX_INDEX);
    return SceneError::OK;
}

SceneError InNodeTriLoaderIndexed::PrimitiveDataCount(size_t& result, PrimitiveDataType primitiveDataType) const
{
    int posIndex = static_cast<int>(primitiveDataType);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    switch(primitiveDataType)
    {
        case PrimitiveDataType::POSITION:
        case PrimitiveDataType::NORMAL:
        case PrimitiveDataType::UV:
            result = node.CommonListSize(positionName);
            break;
        case PrimitiveDataType::VERTEX_INDEX:
            result = node.AccessListTotalCount(positionName);
            break;
        default:
            return SceneError::SURFACE_LOADER_INTERNAL_ERROR;
    }
    return SceneError::OK;
}

SceneError InNodeTriLoaderIndexed::PrimDataLayout(PrimitiveDataLayout& result,
                                                  PrimitiveDataType primitiveDataType) const
{
    if(primitiveDataType == PrimitiveDataType::POSITION ||
       primitiveDataType == PrimitiveDataType::NORMAL)
        result = PrimitiveDataLayout::FLOAT_3;
    else if(primitiveDataType == PrimitiveDataType::UV)
        result = PrimitiveDataLayout::FLOAT_2;
    else if(primitiveDataType == PrimitiveDataType::VERTEX_INDEX)
        result = PrimitiveDataLayout::UINT64_1;
    else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
    return SceneError::OK;
}

//===============//
// SPHERE LOADER //
//===============//
InNodeSphrLoader::InNodeSphrLoader(const SceneNodeI& node, double time)
    : SurfaceLoader(node, time)
{}

const char* InNodeSphrLoader::SufaceDataFileExt() const
{
    return TypeName();
}

SceneError InNodeSphrLoader::AABB(std::vector<AABB3>& result) const
{
    result.resize(node.IdCount());
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

SceneError InNodeSphrLoader::PrimitiveRanges(std::vector<Vector2ul>& result) const
{
    result.resize(node.IdCount());
    for(size_t i = 0; i < node.IdCount(); i++)
    {
        result[i] = Vector2ul(i, i + 1);
    }
    return SceneError::OK;
}

SceneError InNodeSphrLoader::PrimitiveCounts(std::vector<size_t>& result) const
{
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    size_t primDataCount = node.IdCount();

    result.resize(primDataCount);
    std::fill_n(result.begin(), primDataCount, 1);
    return SceneError::OK;
}

SceneError InNodeSphrLoader::PrimitiveDataRanges(std::vector<Vector2ul>& result) const
{
    return PrimitiveRanges(result);
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

SceneError InNodeSphrLoader::HasPrimitiveData(bool& r, PrimitiveDataType primitiveDataType) const
{
    r = (primitiveDataType == PrimitiveDataType::POSITION ||
         primitiveDataType == PrimitiveDataType::RADIUS);
    return SceneError::OK;
}

SceneError InNodeSphrLoader::PrimitiveDataCount(size_t& result, PrimitiveDataType) const
{
    SceneError e = SceneError::OK;
    std::vector<size_t> primCounts;
    if((e = PrimitiveCounts(primCounts)) != SceneError::OK)
        return e;

    result = std::accumulate(primCounts.cbegin(), primCounts.cend(), 0ull);
    return e;
}

SceneError InNodeSphrLoader::PrimDataLayout(PrimitiveDataLayout& result, PrimitiveDataType primitiveDataType) const
{
    for(size_t i = 0; i < node.IdCount(); i++)
    {
        if(primitiveDataType == PrimitiveDataType::POSITION)
            result = PrimitiveDataLayout::FLOAT_3;
        else if(primitiveDataType == PrimitiveDataType::RADIUS)
            result = PrimitiveDataLayout::FLOAT_1;
        else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
    }
    return SceneError::OK;
}