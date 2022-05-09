#include "GFGSurfaceLoaderPool.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

// Surface Loaders
#include "GFGSurfaceLoader.h"

namespace TypeGenWrappers
{
    // Template Type Gen Wrapper
    template <class T>
    SurfaceLoaderI* GFGSurfaceLoaderConstruct(const std::string& scenePath,
                                              const std::string& fileExt,
                                              const std::string_view loggerName,
                                              const SceneNodeI& node,
                                              double time)
    {
        return new T(scenePath, fileExt, loggerName, node, time);
    }
}

GFGSurfaceLoaderPool::GFGSurfaceLoaderPool()
{
    using namespace std::string_literals;

    // Start Logging
    try
    {
        auto logger = spdlog::basic_logger_mt(GFGLogName.data(),
                                              GFGLogFileName.data(),
                                              true);
    }
    catch(const spdlog::spdlog_ex& ex)
    {
        METU_ERROR_LOG("Log init failed: {:S}", ex.what());
    }

    // Only one generator is available for gfg
    gfgSLGenerators.emplace_back(std::string(GFGFileExt),
                                 GFGLogName,
                                 TypeGenWrappers::GFGSurfaceLoaderConstruct<GFGSurfaceLoader>,
                                 TypeGenWrappers::DefaultDestruct<SurfaceLoaderI>);
    surfaceLoaderGenerators.emplace(std::string(GFGPrefix) + GFGFileExt.data(),
                                    &gfgSLGenerators.back());
}

GFGSurfaceLoaderPool::~GFGSurfaceLoaderPool()
{
    // Destroy Logger
    spdlog::drop(GFGLogName.data());
}