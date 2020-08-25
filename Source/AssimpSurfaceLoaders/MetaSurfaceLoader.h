#pragma once

#include "RayLib/SurfaceLoaderI.h"
#include "RayLib/SceneNodeI.h"

#include "AssimpForward.h"

#include <string>

class AssimpMetaSurfaceLoader : public SurfaceLoader
{
    public:
        static constexpr const char* InnerIdJSON    = "innerIndex";

    private:
        Assimp::Importer&                           importer;
        const aiScene*                              scene;
        const std::string&                          extension;

        // Inner Ids
        const UIntList                              innerIds;

    protected:
    public:
        // Constructors & Destructor    
                                    AssimpMetaSurfaceLoader(Assimp::Importer&,                                                         
                                                            const std::string& scenePath,
                                                            const std::string& fileExt,
                                                            const SceneNodeI& node, double time = 0.0);
                                    AssimpMetaSurfaceLoader(const AssimpMetaSurfaceLoader&) = delete;
        AssimpMetaSurfaceLoader&    operator=(const AssimpMetaSurfaceLoader&) = delete;
                                    ~AssimpMetaSurfaceLoader();

        // Interface
        const char*                 SufaceDataFileExt() const override;
        // Per Batch Fetch
        SceneError                  AABB(std::vector<AABB3>&) const override;
        SceneError                  PrimitiveRanges(std::vector<Vector2ul>&) const override;
        SceneError                  PrimitiveCounts(std::vector<size_t>&) const override;
        SceneError                  PrimitiveDataRanges(std::vector<Vector2ul>&) const override;        
        // Entire Data Fetch
        SceneError                  GetPrimitiveData(Byte*, PrimitiveDataType primitiveDataType) const override;
        SceneError                  HasPrimitiveData(bool&, PrimitiveDataType) const override;
        SceneError                  PrimitiveDataCount(size_t&, PrimitiveDataType primitiveDataType) const override;
        SceneError                  PrimDataLayout(PrimitiveDataLayout&,
                                                   PrimitiveDataType primitiveDataType) const override;
};