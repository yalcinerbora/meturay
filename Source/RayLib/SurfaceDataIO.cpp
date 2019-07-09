#include "SurfaceDataIO.h"
#include "SceneIO.h"
#include "Sphere.h"
#include "Triangle.h"
#include "PrimitiveDataTypes.h"
#include "SceneNodeI.h"

#include <numeric>

class InNodeTriLoader : public SurfaceDataLoader
{
    private:
    protected:
    public:
        // Constructors & Destructor
                            InNodeTriLoader(const SceneNodeI&, double time = 0.0);
                            ~InNodeTriLoader() = default;

     // Type Determination
     const char*            SufaceDataFileExt() const override;
     SceneError             PrimDataLayout(PrimitiveDataLayout*,
                                           PrimitiveDataType primitiveDataType) const override;
     SceneError             DataOffsets(size_t*, PrimitiveDataType primitiveDataType) const override;
     SceneError             PrimitiveCount(size_t*) const override;
     SceneError             AABB(AABB3*) const override;
     SceneError             GetPrimitiveData(Byte*, PrimitiveDataType primitiveDataType) const override;
};

InNodeTriLoader::InNodeTriLoader(const SceneNodeI& node, double time)
    : SurfaceDataLoader(node, time)
{
    if(node.IdCount() != 0) throw SceneException(SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR,
                                                 "InNodeTriangle cannot have multi id node.");
}

const char* InNodeTriLoader::SufaceDataFileExt() const
{
    return NodeTriangleName;
}

SceneError InNodeTriLoader::PrimDataLayout(PrimitiveDataLayout* result, PrimitiveDataType primitiveDataType) const
{
    if(primitiveDataType == PrimitiveDataType::POSITION ||
        primitiveDataType == PrimitiveDataType::NORMAL)
        result[0] = PrimitiveDataLayout::FLOAT_3;
    else if(primitiveDataType == PrimitiveDataType::UV)
        result[0] = PrimitiveDataLayout::FLOAT_1;
    else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
}

SceneError InNodeTriLoader::DataOffsets(size_t* result, PrimitiveDataType primitiveDataType) const
{
    result[0] = 0;
    return SceneError::OK;
}

SceneError InNodeTriLoader::PrimitiveCount(size_t* result) const
{
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];

    size_t count = node.AccessCount(positionName);
    if(count % 3 != 0) return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;
    result[0] = count;
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
    for(size_t i = 0; positions.size() / 3; i++)
    {
        result[0].Union(Triangle::BoundingBox(positions[i * 3 + 0],
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

class InNodeSphrLoader : public SurfaceDataLoader
{
    private:
    protected:
    public:
        // Constructors & Destructor
                                InNodeSphrLoader(const SceneNodeI& node, double time = 0.0);
                                ~InNodeSphrLoader() = default;

        // Type Determination
        const char*             SufaceDataFileExt() const override;
        SceneError              PrimDataLayout(PrimitiveDataLayout*,
                                               PrimitiveDataType primitiveDataType) const override;
        SceneError              DataOffsets(size_t*, PrimitiveDataType primitiveDataType) const override;
        SceneError              PrimitiveCount(size_t*) const override;
        SceneError              AABB(AABB3*) const override;
        SceneError              GetPrimitiveData(Byte*, PrimitiveDataType primitiveDataType) const override;
};

InNodeSphrLoader::InNodeSphrLoader(const SceneNodeI&, double time)
    : SurfaceDataLoader(node, time)
{}

// Type Determination
const char* InNodeSphrLoader::SufaceDataFileExt() const
{
    return NodeSphereName;
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
}

SceneError InNodeSphrLoader::DataOffsets(size_t* result, PrimitiveDataType primitiveDataType) const
{
    std::iota(result, result + node.IdCount(), 0);
    return SceneError::OK;
}

SceneError InNodeSphrLoader::PrimitiveCount(size_t* result) const
{
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    size_t primDataCount = node.AccessCount(positionName);

    std::fill_n(result, primDataCount, 1);
    return SceneError::OK;
}

SceneError InNodeSphrLoader::AABB(AABB3* result) const
{
    int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
    const int radIndex = static_cast<int>(PrimitiveDataType::RADIUS);
    const std::string positionName = PrimitiveDataTypeNames[posIndex];
    const std::string radiusName = PrimitiveDataTypeNames[radIndex];

    std::vector<Vector3> positions = node.AccessVector3List(positionName, time);
    std::vector<float> radiuses = node.AccessFloatList(radiusName, time);

    if(positions.size() != node.IdCount() || radiuses.size() != node.IdCount())
        return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

    for(size_t i = 0; i < node.IdCount(); i++)
    {
        result[i] = Sphere::BoundingBox(positions[i], radiuses[i]);
    }
}

SceneError InNodeSphrLoader::GetPrimitiveData(Byte* memory, PrimitiveDataType primitiveDataType) const
{
    int nameIndex = static_cast<int>(primitiveDataType);
    const std::string name = PrimitiveDataTypeNames[nameIndex];

    if(primitiveDataType == PrimitiveDataType::POSITION)
    {
        std::vector<Vector3> result = node.AccessVector3List(name, time);
        std::copy(result.begin(), result.end(), reinterpret_cast<Vector3*>(memory));
        return SceneError::OK;
    }
    else if(primitiveDataType == PrimitiveDataType::RADIUS)
    {
        std::vector<float> result = node.AccessFloatList(name, time);
        std::copy(result.begin(), result.end(), reinterpret_cast<float*>(memory));
        return SceneError::OK;
    }
    else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
}

std::unique_ptr<SurfaceDataLoaderI> SurfaceDataIO::GenSurfaceDataLoader(const SceneNodeI& properties,
                                                                        double time)
{
    const std::string name = properties.AccessString(SceneIO::NAME);
    const std::string ext = SceneIO::StripFileExt(name);

    // There shoudl
    if(ext == NodeSphereName)
    {
        SurfaceDataLoaderI* loader = new InNodeSphrLoader(properties, time);
        return std::unique_ptr<SurfaceDataLoaderI>(loader);
    }
    else if(ext == NodeTriangleName)
    {
        SurfaceDataLoaderI* loader = new InNodeTriLoader(properties, time);
        return std::unique_ptr<SurfaceDataLoaderI>(loader);
    }
    else throw SceneException(SceneError::NO_LOGIC_FOR_SURFACE_DATA);
}