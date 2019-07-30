#include "SurfaceLoaderGenerator.h"
#include "SceneError.h"
#include "SceneNodeI.h"
#include "SceneIO.h"

// Loaders
#include "BasicSurfaceLoaders.h"

// SurfaceLoader Constructor/Destructor for pre-loadede DLL config
template <class Base, class Loader>
Base* SurfaceLoaderConstruct(const SceneNodeI& properties,
                             double time)
{
    return new Loader(properties, time);
}

template <class T>
void SurfaceLoaderDestruct(T* t)
{
    if(t) delete t;
}

SurfaceLoaderGenerator::SurfaceLoaderGenerator()
{
    loaderGenerators.emplace(InNodeTriLoader::TypeName(),
                             SurfaceLoaderGen(SurfaceLoaderConstruct<SurfaceLoaderI, InNodeTriLoader>,
                                              SurfaceLoaderDestruct<SurfaceLoaderI>));
    loaderGenerators.emplace(InNodeTriLoaderIndexed::TypeName(),
                             SurfaceLoaderGen(SurfaceLoaderConstruct<SurfaceLoaderI, InNodeTriLoaderIndexed>,
                                              SurfaceLoaderDestruct<SurfaceLoaderI>));
    loaderGenerators.emplace(InNodeSphrLoader::TypeName(),
                             SurfaceLoaderGen(SurfaceLoaderConstruct<SurfaceLoaderI, InNodeSphrLoader>,
                                              SurfaceLoaderDestruct<SurfaceLoaderI>));
}

// Interface
SceneError SurfaceLoaderGenerator::GenerateSurfaceLoader(SharedLibPtr<SurfaceLoaderI>& ptr,
                                                         const SceneNodeI& properties,
                                                         double time) const
{
    const std::string name = properties.Name();
    const std::string tag = properties.Tag();
    const std::string ext = SceneIO::StripFileExt(name);

    // TODO: Add custom suffix from node
    const std::string mangledName = ext + tag;

    // Cannot Find Already Constructed Type
    // Generate
    auto loc = loaderGenerators.find(mangledName);
    if(loc == loaderGenerators.end())
        return SceneError::NO_LOGIC_FOR_SURFACE_DATA;

    ptr = loc->second(properties, time);
    return SceneError::OK;
}

SceneError SurfaceLoaderGenerator::IncludeLoadersFromDLL(const SharedLib&,
                                                         const std::string& mangledName) const
{
    // TODO: do this
    return SceneError::OK;
}