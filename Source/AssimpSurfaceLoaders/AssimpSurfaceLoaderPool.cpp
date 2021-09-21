#include "AssimpSurfaceLoaderPool.h"

#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/postprocess.h>
// Surface Loaders
#include "MetaSurfaceLoader.h"

namespace TypeGenWrappers
{
    // Template Type Gen Wrapper
    template <class T>
    SurfaceLoaderI* AssimpSurfaceLoaderConstruct(Assimp::Importer& importer,
                                                 const std::string& scenePath,
                                                 const std::string& fileExt,
                                                 const SceneNodeI& node,
                                                 double time)
    {
        return new T(importer, scenePath, fileExt, node, time);
    }
}

AssimpSurfaceLoaderPool::AssimpSurfaceLoaderPool()
{
    static_assert(sizeof(aiVector3D) == sizeof(Vector3), "assimp Vector3 Vector3 mismatch");
    static_assert(sizeof(aiVector2D) == sizeof(Vector2), "assimp Vector2 Vector2 mismatch");

    // Start Logging
    Assimp::DefaultLogger::create(AssimpLogFileName.data(),
                                  Assimp::Logger::VERBOSE);
                                  //,
                                  //aiDefaultLogStream_STDOUT);

    // Do not import line or points
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE,
                                aiPrimitiveType_POINT | aiPrimitiveType_LINE);
    // Import only position, normal, tangent (if provided) and uv
    importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,
                                aiComponent_BONEWEIGHTS |
                                aiComponent_COLORS);

    // First Generate Surface Loader Generators
    // Convenience Lambda
    auto GenSurfaceLoader = [&](const std::string& extension)
    {
        assimpSLGenerators.emplace_back(importer,
                                        extension,
                                        TypeGenWrappers::AssimpSurfaceLoaderConstruct<AssimpMetaSurfaceLoader>,
                                        TypeGenWrappers::DefaultDestruct<SurfaceLoaderI>);
        surfaceLoaderGenerators.emplace(std::string(AssimpPrefix) + extension,
                                        &assimpSLGenerators.back());
    };

    // Add Surface Loaders
    using namespace std::string_literals;

    // Some basic stuff to use assimp for
    GenSurfaceLoader("obj"s);
    GenSurfaceLoader("fbx"s);
    GenSurfaceLoader("blend"s);
    GenSurfaceLoader("ply"s);
    GenSurfaceLoader("off"s);
}

AssimpSurfaceLoaderPool::~AssimpSurfaceLoaderPool()
{
    // Destroy Logger
    Assimp::DefaultLogger::kill();
}