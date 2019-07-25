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
     const char*                    SufaceDataFileExt() const override;     
     SceneError                     BatchOffsets(size_t*) const override;
     SceneError                     PrimitiveCounts(size_t*) const override;
     SceneError                     PrimDataLayout(PrimitiveDataLayout*,
                                                   PrimitiveDataType primitiveDataType) const override;

     SceneError                     AABB(AABB3*) const override;
     SceneError                     GetPrimitiveData(Byte*, PrimitiveDataType primitiveDataType) const override;
     SceneError                     PrimitiveDataCount(size_t*, PrimitiveDataType primitiveDataType) const override;
};

class InNodeSphrLoader : public SurfaceLoader
{
    public:
        static constexpr const char*    TypeName() { return "nodeSphere"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                        InNodeSphrLoader(const SceneNodeI& node, double time = 0.0);
                                        ~InNodeSphrLoader() = default;

        // Type Determination
         const char*                    SufaceDataFileExt() const override;     
         SceneError                     BatchOffsets(size_t*) const override;
         SceneError                     PrimitiveCounts(size_t*) const override;
         SceneError                     PrimDataLayout(PrimitiveDataLayout*,
                                                       PrimitiveDataType primitiveDataType) const override;

         SceneError                     AABB(AABB3*) const override;
         SceneError                     GetPrimitiveData(Byte*, PrimitiveDataType primitiveDataType) const override;
         SceneError                     PrimitiveDataCount(size_t*, PrimitiveDataType primitiveDataType) const override;
};