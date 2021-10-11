#pragma once

#include <memory>
#include <vector>

#include "SurfaceLoaderI.h"
#include "Vector.h"

class InNodeTriLoader : public SurfaceLoader
{
    public:
        static constexpr const char* TypeName() { return "nodeTriangle"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                        InNodeTriLoader(const SceneNodeI&, double time = 0.0);
                        ~InNodeTriLoader() = default;

        // Type Determination
        const char*     SufaceDataFileExt() const override;

        // Per Batch Fetch
        SceneError      AABB(std::vector<AABB3>&) const override;
        SceneError      PrimitiveRanges(std::vector<Vector2ul>&) const override;
        SceneError      PrimitiveCounts(std::vector<size_t>&) const override;
        SceneError      PrimitiveDataRanges(std::vector<Vector2ul>&) const override;

        // Entire Data Fetch
        SceneError      GetPrimitiveData(Byte*, PrimitiveDataType primitiveDataType) const override;
        SceneError      HasPrimitiveData(bool&, PrimitiveDataType) const override;
        SceneError      PrimitiveDataCount(size_t&, PrimitiveDataType primitiveDataType) const override;
        SceneError      PrimDataLayout(PrimitiveDataLayout&,
                                       PrimitiveDataType primitiveDataType) const override;
};

class InNodeTriLoaderIndexed : public SurfaceLoader
{
    public:
        static constexpr const char* TypeName() { return "nodeTriangleIndexed"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                        InNodeTriLoaderIndexed(const SceneNodeI&, double time = 0.0);
                        ~InNodeTriLoaderIndexed() = default;

        // Type Determination
        const char*     SufaceDataFileExt() const override;

        // Per Batch Fetch
        SceneError      AABB(std::vector<AABB3>&) const override;
        SceneError      PrimitiveRanges(std::vector<Vector2ul>&) const override;
        SceneError      PrimitiveCounts(std::vector<size_t>&) const override;
        SceneError      PrimitiveDataRanges(std::vector<Vector2ul>&) const override;

        // Entire Data Fetch
        SceneError      GetPrimitiveData(Byte*, PrimitiveDataType primitiveDataType) const override;
        SceneError      HasPrimitiveData(bool&, PrimitiveDataType) const override;
        SceneError      PrimitiveDataCount(size_t&, PrimitiveDataType primitiveDataType) const override;
        SceneError      PrimDataLayout(PrimitiveDataLayout&,
                                       PrimitiveDataType primitiveDataType) const override;
};

class InNodeSphrLoader : public SurfaceLoader
{
    public:
        static constexpr const char* TypeName() { return "nodeSphere"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                        InNodeSphrLoader(const SceneNodeI& node, double time = 0.0);
                        ~InNodeSphrLoader() = default;

        // Type Determination
         const char*    SufaceDataFileExt() const override;

        // Per Batch Fetch
        SceneError      AABB(std::vector<AABB3>&) const override;
        SceneError      PrimitiveRanges(std::vector<Vector2ul>&) const override;
        SceneError      PrimitiveCounts(std::vector<size_t>&) const override;
        SceneError      PrimitiveDataRanges(std::vector<Vector2ul>&) const override;

        // Entire Data Fetch
        SceneError      GetPrimitiveData(Byte*, PrimitiveDataType primitiveDataType) const override;
        SceneError      HasPrimitiveData(bool&, PrimitiveDataType) const override;
        SceneError      PrimitiveDataCount(size_t&, PrimitiveDataType primitiveDataType) const override;
        SceneError      PrimDataLayout(PrimitiveDataLayout&,
                                       PrimitiveDataType primitiveDataType) const override;
};