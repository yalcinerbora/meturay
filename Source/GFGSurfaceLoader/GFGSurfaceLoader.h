#pragma once

#include "RayLib/SurfaceLoaderI.h"
#include "RayLib/SceneNodeI.h"

#include <string>
#include <fstream>

class GFGFileLoader;
class GFGFileReaderSTL;

class GFGSurfaceLoader : public SurfaceLoader
{
    public:
        static constexpr const char* InnerIdJSON    = "innerIndex";

    private:
        const std::string                   extension;
        const std::string                   filePath;
        const std::string_view              loggerName;

        std::ifstream                       file;
        std::unique_ptr<GFGFileReaderSTL>   gfgFileReaderSTL;
        std::unique_ptr<GFGFileLoader>      gfgFile;

        // Inner Ids
        const UIntList                      innerIds;
        // Error Logging
        void                                LogError(const std::string&);

    protected:
    public:
        // Constructors & Destructor
                                    GFGSurfaceLoader(const std::string& scenePath,
                                                     const std::string& fileExt,
                                                     const std::string_view loggerName,
                                                     const SceneNodeI& node, double time = 0.0);
                                    GFGSurfaceLoader(const GFGSurfaceLoader&) = delete;
        GFGSurfaceLoader&           operator=(const GFGSurfaceLoader&) = delete;
                                    ~GFGSurfaceLoader() = default;

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