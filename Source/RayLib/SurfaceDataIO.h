#pragma once

#include <memory>
#include <vector>
#include <nlohmann/json_fwd.hpp>

#include "RayLib/Vector.h"
#include "RayLib/AABB.h"
#include "RayLib/Types.h"

struct SceneError;
class SceneNodeI;

enum class PrimitiveDataLayout : uint32_t;
enum class PrimitiveDataType;

constexpr const char* NodeSphereName = "nodeSphere";
constexpr const char* NodeTriangleName = "nodeTriangle";

class SurfaceDataLoaderI
{
    public:
        virtual                         ~SurfaceDataLoaderI() = default;

        // Type Determination
        virtual const SceneNodeI&       SceneNode() const = 0;
        virtual const char*             SufaceDataFileExt() const = 0;
        virtual SceneError              BatchOffsets(size_t*) const = 0;
        virtual SceneError              PrimitiveCount(size_t*) const = 0;
        virtual SceneError              PrimDataLayout(PrimitiveDataLayout*,
                                                       PrimitiveDataType primitiveDataType) const = 0;
        virtual SceneError              AABB(AABB3*) const = 0;        
        virtual SceneError              GetPrimitiveData(Byte*, PrimitiveDataType primitiveDataType) const = 0;
};

class SurfaceDataLoader : public SurfaceDataLoaderI
{
    private:
    protected:
        const SceneNodeI&           node;
        double                      time;

        // Constructor
                                    SurfaceDataLoader(const SceneNodeI& node,
                                                      double time = 0.0);

    public:
        // Destructor
                                    ~SurfaceDataLoader() = default;
        // Implementation
        const SceneNodeI&           SceneNode() const override;
};

inline SurfaceDataLoader::SurfaceDataLoader(const SceneNodeI& node, double time)
    : node(node)
    , time(time)
{}

inline const SceneNodeI& SurfaceDataLoader::SceneNode() const
{
    return node;
}

namespace SurfaceDataIO
{
    std::unique_ptr<SurfaceDataLoaderI> GenSurfaceDataLoader(const SceneNodeI& properties,
                                                             double time);
}

using SurfaceDataLoaders = std::vector<std::unique_ptr<SurfaceDataLoaderI>>;