#pragma once

#include <vector>

#include "AABB.h"
#include "Types.h"

struct SceneError;
class SceneNodeI;

enum class PrimitiveDataLayout : uint32_t;
enum class PrimitiveDataType;

// Node Specific Loader
class SurfaceLoaderI
{
    public:
        virtual                         ~SurfaceLoaderI() = default;

        // Type Determination
        virtual const SceneNodeI&       SceneNode() const = 0;
        virtual const char*             SufaceDataFileExt() const = 0;

        // Per Batch Fetch
        virtual SceneError              AABB(std::vector<AABB3>&) const = 0;
        virtual SceneError              PrimitiveRanges(std::vector<Vector2ul>&) const = 0;
        virtual SceneError              PrimitiveCounts(std::vector<size_t>&) const = 0;
        virtual SceneError              PrimitiveDataRanges(std::vector<Vector2ul>&) const = 0;

        // Entire Data Fetch
        virtual SceneError              GetPrimitiveData(Byte*, PrimitiveDataType primitiveDataType) const = 0;
        virtual SceneError              HasPrimitiveData(bool&, PrimitiveDataType) const = 0;
        virtual SceneError              PrimitiveDataCount(size_t&, PrimitiveDataType primitiveDataType) const = 0;
        virtual SceneError              PrimDataLayout(PrimitiveDataLayout&,
                                                       PrimitiveDataType primitiveDataType) const = 0;
};

class SurfaceLoader : public SurfaceLoaderI
{
    private:
    protected:
        const SceneNodeI&           node;
        double                      time;

        // Constructor
                                    SurfaceLoader(const SceneNodeI& node,
                                                  double time = 0.0);

    public:
        // Destructor
                                    ~SurfaceLoader() = default;
        // Implementation
        const SceneNodeI&           SceneNode() const override;
};

inline SurfaceLoader::SurfaceLoader(const SceneNodeI& node, double time)
    : node(node)
    , time(time)
{}

inline const SceneNodeI& SurfaceLoader::SceneNode() const
{
    return node;
}
