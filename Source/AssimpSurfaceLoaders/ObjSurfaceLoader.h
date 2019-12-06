#pragma once

#include "RayLib/SurfaceLoaderI.h"
#include "AssimpForward.h"

// Loaders
#include "ObjSurfaceLoader.h"
#include <string>

class ObjSurfaceLoader : public SurfaceLoader
{
    public:
        static constexpr const char* Extension() { return "obj"; }
        static constexpr const char* TypeName() { return "assimp_obj"; }

    private:
        Assimp::Importer&   importer;
        const aiScene*      scene;

    protected:
    public:
        // Constructors & Destructor    
                            ObjSurfaceLoader(Assimp::Importer&, const std::string& scenePath,
                                             const SceneNodeI& node, double time = 0.0);
                            ObjSurfaceLoader(const ObjSurfaceLoader&) = delete;
                            ObjSurfaceLoader& operator=(const ObjSurfaceLoader&) = delete;
                            ~ObjSurfaceLoader();

        // Interface
        const char*         SufaceDataFileExt() const override;
        // Per Batch Fetch
        SceneError          AABB(std::vector<AABB3>&) const override;
        SceneError          PrimitiveRanges(std::vector<Vector2ul>&) const override;
        SceneError          PrimitiveCounts(std::vector<size_t>&) const override;
        SceneError          PrimitiveDataRanges(std::vector<Vector2ul>&) const override;        
        // Entire Data Fetch
        SceneError          GetPrimitiveData(Byte*, PrimitiveDataType primitiveDataType) const override;
        SceneError          PrimitiveDataCount(size_t&, PrimitiveDataType primitiveDataType) const override;
        SceneError          PrimDataLayout(PrimitiveDataLayout&,
                                           PrimitiveDataType primitiveDataType) const override;
};
