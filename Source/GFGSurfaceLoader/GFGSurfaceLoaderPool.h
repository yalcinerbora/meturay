#pragma once

#include "RayLib/SurfaceLoaderPoolI.h"
#include <list>

using GFGSurfaceLoaderGeneratorFunc = SurfaceLoaderI* (*)(const std::string& scenePath,
                                                          const std::string& fileExt,
                                                          const std::string_view loggerName,
                                                          const SceneNodeI&,
                                                          double time);

class GFGSurfaceLoaderGen : public SurfaceLoaderGenI
{
    private:
        const std::string                                   fileExt;
        const std::string_view                              loggerName;
        GFGSurfaceLoaderGeneratorFunc                       gFunc;
        ObjDestroyerFunc<SurfaceLoaderI>                    dFunc;

    public:
        // Constructor & Destructor
        GFGSurfaceLoaderGen(const std::string& fileExt,
                            const std::string_view loggerName,
                            GFGSurfaceLoaderGeneratorFunc g,
                            ObjDestroyerFunc<SurfaceLoaderI> d)
            : fileExt(fileExt)
            , loggerName(loggerName)
            , gFunc(g)
            , dFunc(d)
        {}

        SurfaceLoaderPtr operator()(const std::string& scenePath,
                                    const SceneNodeI& n,
                                    double time) const override
        {
            SurfaceLoaderI* loader = gFunc(scenePath, fileExt, loggerName, n, time);
            return SurfaceLoaderPtr(loader, dFunc);
        }
};

class GFGSurfaceLoaderPool : public SurfaceLoaderPoolI
{
    private:
        static constexpr std::string_view    GFGFileExt         = "gfg";
        static constexpr std::string_view    GFGPrefix          = "gfg_";
        static constexpr std::string_view    GFGLogFileName     = "gfg_log";
        static constexpr std::string_view    GFGLogName         = "GFGLogger";

        // Single Meta Surface loader for known file types
        std::list<GFGSurfaceLoaderGen>      gfgSLGenerators;

    protected:
    public:
        // Constructors & Destructor
                                GFGSurfaceLoaderPool();
                                GFGSurfaceLoaderPool(const GFGSurfaceLoaderPool&) = delete;
        GFGSurfaceLoaderPool&   operator=(const GFGSurfaceLoaderPool&) = delete;
                                ~GFGSurfaceLoaderPool();
};