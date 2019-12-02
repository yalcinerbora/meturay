#include "SurfaceLoaderGenerator.h"
#include "SceneError.h"
#include "SceneNodeI.h"
#include "SceneIO.h"

// Loaders
#include "BasicSurfaceLoaders.h"

// SurfaceLoader Constructor/Destructor for pre-loadede DLL config
template <class Base, class Loader>
Base* InNodeSurfaceLoaderConstruct(const std::string& scenePath,
                                   const SceneNodeI& properties,
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
                             SurfaceLoaderGen(InNodeSurfaceLoaderConstruct<SurfaceLoaderI, InNodeTriLoader>,
                                              SurfaceLoaderDestruct<SurfaceLoaderI>));
    loaderGenerators.emplace(InNodeTriLoaderIndexed::TypeName(),
                             SurfaceLoaderGen(InNodeSurfaceLoaderConstruct<SurfaceLoaderI, InNodeTriLoaderIndexed>,
                                              SurfaceLoaderDestruct<SurfaceLoaderI>));
    loaderGenerators.emplace(InNodeSphrLoader::TypeName(),
                             SurfaceLoaderGen(InNodeSurfaceLoaderConstruct<SurfaceLoaderI, InNodeSphrLoader>,
                                              SurfaceLoaderDestruct<SurfaceLoaderI>));
}

// Interface
SceneError SurfaceLoaderGenerator::GenerateSurfaceLoader(SharedLibPtr<SurfaceLoaderI>& ptr,
                                                         const std::string& scenePath,
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

    ptr = loc->second(scenePath, properties, time);
    return SceneError::OK;
}

DLLError SurfaceLoaderGenerator::IncludeLoadersFromDLL(const std::string& libName,
                                                       const std::string& regex,
                                                       const SharedLibArgs& mangledName)
{
    // Find Shared Lib
    SharedLib* libOut;
    auto it = openedLibs.end();
    if((it = openedLibs.find(libName)) != openedLibs.end())
    {
        libOut = &it->second;
    }
    else
    {
        try
        {
            auto it = openedLibs.emplace(libName, SharedLib(libName));
            libOut = &it.first->second;
        }
        catch(const DLLException & e)
        {
            return e;
        }
    }

    PoolKey libKey = {libOut, mangledName};
    SurfaceLoaderPoolPtr* pool;

    // Then Find Pool
    DLLError e = DLLError::OK;
    auto loc = generatedPools.end();
    if((loc = generatedPools.find(libKey)) != generatedPools.end())
    {
        pool = &loc->second;
    }
    else
    {
        SurfaceLoaderPoolPtr ptr = {nullptr, nullptr};
        e = libKey.first->GenerateObject<SurfaceLoaderPoolI>(ptr, libKey.second);
        if(e != DLLError::OK) return e;
        auto it = generatedPools.emplace(libKey, std::move(ptr));
        pool = &(it.first->second);
    }

    const auto newLoaders = (*pool)->SurfaceLoaders(regex);
    loaderGenerators.insert(newLoaders.cbegin(), newLoaders.cend());
    return DLLError::OK;
}