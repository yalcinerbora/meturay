#include "AssimpSurfaceLoaderPool.h"

#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>

AssimpSurfaceLoaderPool::AssimpSurfaceLoaderPool()
{
    // Start Logging
    Assimp::DefaultLogger::create("", Assimp::Logger::VERBOSE,
                                  aiDefaultLogStream_STDOUT);

    // Add Surface Loader Generators to the Pool
    // ...........
}

AssimpSurfaceLoaderPool::~AssimpSurfaceLoaderPool()
{
    // Destroy Logger
    Assimp::DefaultLogger::kill();
}