#pragma once
/**

Base implementation of Logic Generator

Provides storage of types (via across boundary capable unique ptrs)
and it adds default accelerators and primitives as default types.

*/

#include <map>
#include <list>

#include "RayLib/Types.h"
#include "RayLib/SharedLib.h"

#include "TracerLogicGeneratorI.h"
#include "TracerTypeGenerators.h"
#include "TracerLogicPools.h"

using NameGPUPair = std::pair<std::string, const CudaGPU*>;

using AcceleratorPoolPtr = SharedLibPtr<AcceleratorLogicPoolI>;
using MaterialPoolPtr = SharedLibPtr<MaterialLogicPoolI>;
using PrimitivePoolPtr = SharedLibPtr<PrimitiveLogicPoolI>;
using BaseAcceleratorPoolPtr = SharedLibPtr<BaseAcceleratorLogicPoolI>;
using TracerPoolPtr = SharedLibPtr<TracerPoolI>;

class TracerLogicGenerator : public TracerLogicGeneratorI
{
    private:
    protected:
        // Shared Libraries That are Loaded
        std::map<std::string, SharedLib>            openedLibs;
        // Included Pools
        std::map<PoolKey, AcceleratorPoolPtr>       loadedAccPools;
        std::map<PoolKey, MaterialPoolPtr>          loadedMatPools;
        std::map<PoolKey, PrimitivePoolPtr>         loadedPrimPools;
        std::map<PoolKey, BaseAcceleratorPoolPtr>   loadedBaseAccPools;
        std::map<PoolKey, TracerPoolPtr>       loadedTracerPools;        
        // All Combined Type Generation Functions
        // Type Generation Functions
        std::map<std::string, GPUPrimGroupGen>      primGroupGenerators;
        std::map<std::string, GPUAccelGroupGen>     accelGroupGenerators;
        std::map<std::string, GPUMatGroupGen>       matGroupGenerators;
        std::map<std::string, GPUBaseAccelGen>      baseAccelGenerators;
        std::map<std::string, GPUTracerGen>         tracerGenerators;
        // Helper Funcs
        uint32_t                                    CalculateHitStructSize();
        DLLError                                    FindOrGenerateSharedLib(SharedLib*& libOut,
                                                                            const std::string& libName);

        template <class T>
        DLLError                                    FindOrGeneratePool(SharedLibPtr<T>*&,
                                                                       std::map<PoolKey, SharedLibPtr<T>>&,
                                                                       const PoolKey& libName);

    public:
        // Constructor & Destructor
                                    TracerLogicGenerator();
                                    TracerLogicGenerator(const TracerLogicGenerator&) = delete;
        TracerLogicGenerator&       operator=(const TracerLogicGenerator&) = delete;
                                    ~TracerLogicGenerator() = default;

        // Pritimive
        SceneError                  GeneratePrimitiveGroup(GPUPrimGPtr&,
                                                           const std::string& primitiveType) override;
        // Accelerator
        SceneError                  GenerateAcceleratorGroup(GPUAccelGPtr&,
                                                             const GPUPrimitiveGroupI&,
                                                             const TransformStruct* t,
                                                             const std::string& accelType) override;
        // Material
        SceneError                  GenerateMaterialGroup(GPUMatGPtr&,
                                                          const CudaGPU&,
                                                          const std::string& materialType) override;
        // Base Accelerator should be fetched after all the stuff is generated
        SceneError                  GenerateBaseAccelerator(GPUBaseAccelPtr&,
                                                            const std::string& accelType) override;
                // Tracer Logic
        SceneError                  GenerateTracer(GPUTracerPtr&,
                                                   const TracerParameters&,
                                                   const GPUSceneI&,
                                                   const std::string& tracerType) override;
   
        // Inclusion Functionality
        // Additionally includes the materials, primitives etc. from other libraries
        DLLError                    IncludeBaseAcceleratorsFromDLL(const std::string& libName,
                                                                   const std::string& regex,
                                                                   const SharedLibArgs& mangledName) override;
        DLLError                    IncludeAcceleratorsFromDLL(const std::string& libName,
                                                               const std::string& regex,
                                                               const SharedLibArgs& mangledName) override;
        DLLError                    IncludeMaterialsFromDLL(const std::string& libName,
                                                            const std::string& regex,
                                                            const SharedLibArgs& mangledName) override;
        DLLError                    IncludePrimitivesFromDLL(const std::string& libName,
                                                            const std::string& regex,
                                                            const SharedLibArgs& mangledName) override;
        DLLError                    IncludeEstimatorsFromDLL(const std::string& libName,
                                                             const std::string& regex,
                                                             const SharedLibArgs& mangledName) override;
        DLLError                    IncludeTracersFromDLL(const std::string& libName,
                                                             const std::string& regex,
                                                             const SharedLibArgs& mangledName) override;
};

template <class T>
DLLError TracerLogicGenerator::FindOrGeneratePool(SharedLibPtr<T>*& pool,
                                                  std::map<PoolKey, SharedLibPtr<T>>& generatedPools,
                                                  const PoolKey& libKey)
{
    DLLError e = DLLError::OK;
    auto loc = generatedPools.end();
    if((loc = generatedPools.find(libKey)) != generatedPools.end())
    {
        pool = &loc->second;
        return e;
    }
    else
    {
        SharedLibPtr<T> ptr = {nullptr, nullptr};
        e = libKey.first->GenerateObject<T>(ptr,libKey.second);
        if(e != DLLError::OK) return e;
        auto it = generatedPools.emplace(libKey, std::move(ptr));
        pool = &(it.first->second);
        return e;
    }
}