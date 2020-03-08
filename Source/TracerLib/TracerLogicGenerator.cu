#include "TracerLogicGenerator.h"

#include "RayLib/DLLError.h"

#include "GPUPrimitiveSphere.h"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveEmpty.h"

#include "GPUAcceleratorLinear.cuh"
#include "GPUAcceleratorBVH.cuh"

#include "GPUMaterialI.h"

// Type to utilize the generated ones
extern template class GPUAccLinearGroup<GPUPrimitiveTriangle>;
extern template class GPUAccLinearGroup<GPUPrimitiveSphere>;

extern template class GPUAccBVHGroup<GPUPrimitiveTriangle>;
extern template class GPUAccBVHGroup<GPUPrimitiveSphere>;

// Typedefs for ease of read
using GPUAccTriLinearGroup = GPUAccLinearGroup<GPUPrimitiveTriangle>;
using GPUAccSphrLinearGroup = GPUAccLinearGroup<GPUPrimitiveSphere>;

using GPUAccTriBVHGroup = GPUAccBVHGroup<GPUPrimitiveTriangle>;
using GPUAccSphrBVHGroup = GPUAccBVHGroup<GPUPrimitiveSphere>;

// Some Instantiations
// Constructors
template GPUPrimitiveGroupI* TypeGenWrappers::DefaultConstruct<GPUPrimitiveGroupI,
                                                               GPUPrimitiveTriangle>();

template GPUAcceleratorGroupI* TypeGenWrappers::AccelGroupConstruct<GPUAcceleratorGroupI,
                                                                    GPUAccTriLinearGroup>(const GPUPrimitiveGroupI&,
                                                                                          const TransformStruct*);

// Destructors
template void TypeGenWrappers::DefaultDestruct(GPUPrimitiveGroupI*);
template void TypeGenWrappers::DefaultDestruct(GPUMaterialGroupI*);
template void TypeGenWrappers::DefaultDestruct(GPUAcceleratorGroupI*);

// Helper Funcs
DLLError TracerLogicGenerator::FindOrGenerateSharedLib(SharedLib*& libOut,
                                                       const std::string& libName)
{
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
        catch(const DLLException& e)
        {
            return e;
        }
    }
    return DLLError::OK;
}

TracerLogicGenerator::TracerLogicGenerator()
{
    using namespace TypeGenWrappers;

    // Primitive Defaults
    primGroupGenerators.emplace(GPUPrimitiveTriangle::TypeName(),
                                GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveTriangle>,
                                                DefaultDestruct<GPUPrimitiveGroupI>));
    primGroupGenerators.emplace(GPUPrimitiveSphere::TypeName(),
                                GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveSphere>,
                                                DefaultDestruct<GPUPrimitiveGroupI>));
    primGroupGenerators.emplace(GPUPrimitiveEmpty::TypeName(),
                                GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveEmpty>,
                                                DefaultDestruct<GPUPrimitiveGroupI>));

    // Accelerator Types
    accelGroupGenerators.emplace(GPUAccTriLinearGroup::TypeName(),
                                 GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccTriLinearGroup>,
                                                  DefaultDestruct<GPUAcceleratorGroupI>));
    accelGroupGenerators.emplace(GPUAccSphrLinearGroup::TypeName(),
                                 GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccSphrLinearGroup>,
                                                  DefaultDestruct<GPUAcceleratorGroupI>));
    accelGroupGenerators.emplace(GPUAccTriBVHGroup::TypeName(),
                                 GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccTriBVHGroup>,
                                                  DefaultDestruct<GPUAcceleratorGroupI>));
    accelGroupGenerators.emplace(GPUAccSphrBVHGroup::TypeName(),
                                 GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccSphrBVHGroup>,
                                                  DefaultDestruct<GPUAcceleratorGroupI>));
    // Base Accelerator
    baseAccelGenerators.emplace(GPUBaseAcceleratorLinear::TypeName(),
                                GPUBaseAccelGen(DefaultConstruct<GPUBaseAcceleratorI, GPUBaseAcceleratorLinear>,
                                                DefaultDestruct<GPUBaseAcceleratorI>));
    baseAccelGenerators.emplace(GPUBaseAcceleratorBVH::TypeName(),
                                GPUBaseAccelGen(DefaultConstruct<GPUBaseAcceleratorI, GPUBaseAcceleratorBVH>,
                                                DefaultDestruct<GPUBaseAcceleratorI>));

    // Default Types are loaded
    // Other Types are strongly tied to base tracer logic
    // i.e. Auxiliary Struct Etc.
}

SceneError TracerLogicGenerator::GeneratePrimitiveGroup(GPUPrimGPtr& pg,
                                                        const std::string& primitiveType)
{
    pg = nullptr;       
    auto loc = primGroupGenerators.find(primitiveType);
    if(loc == primGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_PRIMITIVE;

    GPUPrimGPtr ptr = loc->second();
    pg = std::move(ptr);
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateAcceleratorGroup(GPUAccelGPtr& ag,
                                                          const GPUPrimitiveGroupI& pg,
                                                          const TransformStruct* t,
                                                          const std::string& accelType)
{
    ag = nullptr;
    auto loc = accelGroupGenerators.find(accelType);
    if(loc == accelGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_ACCELERATOR;

    GPUAccelGPtr ptr = loc->second(pg, t);
    ag = std::move(ptr);
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateMaterialGroup(GPUMatGPtr& mg,
                                                       const CudaGPU& gpu,
                                                       const std::string& materialType)
{
    mg = nullptr;
    auto loc = matGroupGenerators.find(materialType);
    if(loc == matGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_MATERIAL;
    mg = std::move(loc->second(gpu));
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateBaseAccelerator(GPUBaseAccelPtr& baseAccel,
                                                         const std::string& accelType)
{
    auto loc = baseAccelGenerators.find(accelType);
    if(loc == baseAccelGenerators.end()) return SceneError::NO_LOGIC_FOR_ACCELERATOR;
    baseAccel = std::move(loc->second());
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateTracer(GPUTracerPtr& tracerPtr,
                                                // Args
                                                const TracerParameters& p,
                                                const GPUSceneI& scene,
                                                // Type
                                                const std::string& tracerType)
{
    auto loc = tracerGenerators.find(tracerType);
    if(loc == tracerGenerators.end()) return SceneError::NO_LOGIC_FOR_TRACER;

    tracerPtr = std::move(loc->second(p, scene));
    return SceneError::OK;
}

DLLError TracerLogicGenerator::IncludeBaseAcceleratorsFromDLL(const std::string& libName,
                                                              const std::string& regex,
                                                              const SharedLibArgs& mangledName)
{
    DLLError e = DLLError::OK;
    SharedLib* lib = nullptr;
    e = FindOrGenerateSharedLib(lib, libName);
    if(e != DLLError::OK) return e;

    BaseAcceleratorPoolPtr* pool = nullptr;
    e = FindOrGeneratePool<BaseAcceleratorLogicPoolI>(pool, loadedBaseAccPools,
                                                      {lib, mangledName});
    if(e != DLLError::OK) return e;

    auto map = (*pool)->BaseAcceleratorGenerators(regex);
    baseAccelGenerators.insert(map.cbegin(), map.cend());
    return e;
    return DLLError::OK;
    // Then 
}

DLLError TracerLogicGenerator::IncludeAcceleratorsFromDLL(const std::string& libName,
                                                          const std::string& regex,
                                                          const SharedLibArgs& mangledName)
{
    DLLError e = DLLError::OK;
    SharedLib* lib = nullptr;
    e = FindOrGenerateSharedLib(lib, libName);
    if(e != DLLError::OK) return e;

    AcceleratorPoolPtr* pool = nullptr;
    e = FindOrGeneratePool<AcceleratorLogicPoolI>(pool, loadedAccPools,
                                                  {lib, mangledName});
    if(e != DLLError::OK) return e;

    auto groupMap = (*pool)->AcceleratorGroupGenerators(regex);
    accelGroupGenerators.insert(groupMap.cbegin(), groupMap.cend());
    return e;
}

DLLError TracerLogicGenerator::IncludeMaterialsFromDLL(const std::string& libName,
                                                       const std::string& regex,
                                                       const SharedLibArgs& mangledName)
{
    DLLError e = DLLError::OK;
    SharedLib* lib = nullptr;
    e = FindOrGenerateSharedLib(lib, libName);
    if(e != DLLError::OK) return e;

    MaterialPoolPtr* pool = nullptr;
    e = FindOrGeneratePool<MaterialLogicPoolI>(pool, loadedMatPools,
                                                {lib, mangledName});
    if(e != DLLError::OK) return e;

    auto groupMap = (*pool)->MaterialGroupGenerators(regex);
    matGroupGenerators.insert(groupMap.cbegin(), groupMap.cend());
    return e;
}

DLLError TracerLogicGenerator::IncludePrimitivesFromDLL(const std::string& libName,
                                                        const std::string& regex,
                                                        const SharedLibArgs& mangledName)
{
    DLLError e = DLLError::OK;
    SharedLib* lib = nullptr;
    e = FindOrGenerateSharedLib(lib, libName);
    if(e != DLLError::OK) return e;

    PrimitivePoolPtr* pool = nullptr;
    e = FindOrGeneratePool<PrimitiveLogicPoolI>(pool, loadedPrimPools,
                                                {lib, mangledName});
    if(e != DLLError::OK) return e;

    auto map = (*pool)->PrimitiveGenerators(regex);
    primGroupGenerators.insert(map.cbegin(), map.cend());
    return e;
}

DLLError TracerLogicGenerator::IncludeTracersFromDLL(const std::string& libName,
                                                     const std::string& regex,
                                                     const SharedLibArgs& mangledName)
{
    DLLError e = DLLError::OK;
    SharedLib* lib = nullptr;
    e = FindOrGenerateSharedLib(lib, libName);
    if(e != DLLError::OK) return e;

    TracerPoolPtr* pool = nullptr;
    e = FindOrGeneratePool<TracerPoolI>(pool, loadedTracerPools,
                                        {lib, mangledName});
    if(e != DLLError::OK) return e;

    auto map = (*pool)->TracerGenerators(regex);
    tracerGenerators.insert(map.cbegin(), map.cend());
    return e;
}