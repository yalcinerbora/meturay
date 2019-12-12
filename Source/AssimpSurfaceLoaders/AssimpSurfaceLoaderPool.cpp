#include "AssimpSurfaceLoaderPool.h"

#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>

// Surface Loaders
#include "ObjSurfaceLoader.h"

namespace TypeGenWrappers
{
    // Template Type Gen Wrapper
    template <class T>
    SurfaceLoaderI* AssimpSurfaceLoaderConstruct(Assimp::Importer& importer,
                                                 const std::string& scenePath,
                                                 const SceneNodeI& node,
                                                 double time)
    {
        return new T(importer, scenePath, node, time);
    }
}

AssimpSurfaceLoaderPool::AssimpSurfaceLoaderPool()
    : objSurfaceLoader(importer,
                       TypeGenWrappers::AssimpSurfaceLoaderConstruct<ObjSurfaceLoader>,
                       TypeGenWrappers::DefaultDestruct<SurfaceLoaderI>)
{
    static_assert(sizeof(aiVector3D) == sizeof(Vector3), "assimp Vector3 Vector3 mismatch");
    static_assert(sizeof(aiVector2D) == sizeof(Vector2), "assimp Vector2 Vector2 mismatch");

    // Start Logging
    Assimp::DefaultLogger::create("", Assimp::Logger::VERBOSE,
                                  aiDefaultLogStream_STDOUT);

    // Add Surface Loader Generators to the Pool
    surfaceLoaderGenerators.emplace(ObjSurfaceLoader::TypeName(),
                                    &objSurfaceLoader);
}

AssimpSurfaceLoaderPool::~AssimpSurfaceLoaderPool()
{
    // Destroy Logger
    Assimp::DefaultLogger::kill();
}