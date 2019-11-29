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
#include "DefaultTypeGenerators.h"
#include "TracerLogicPools.h"

using NameIdPair = std::pair<std::string, int>;

using AcceleratorPoolPtr = SharedLibPtr<AcceleratorLogicPoolI>;
using MaterialPoolPtr = SharedLibPtr<MaterialLogicPoolI>;
using PrimitivePoolPtr = SharedLibPtr<PrimitiveLogicPoolI>;
using BaseAcceleratorPoolPtr = SharedLibPtr<BaseAcceleratorLogicPoolI>;

using PoolKey = std::pair<SharedLib*, SharedLibArgs>;

class TracerLogicGenerator : public TracerLogicGeneratorI
{
    private:
    protected:
        // Shared Libraries That are Loaded
        std::map<std::string, SharedLib>                openedLibs;
        // Included Pools
        std::map<PoolKey, AcceleratorPoolPtr>           loadedAccPools;
        std::map<PoolKey, MaterialPoolPtr>              loadedMatPools;
        std::map<PoolKey, PrimitivePoolPtr>             loadedPrimPools;
        std::map<PoolKey, BaseAcceleratorPoolPtr>       loadedBaseAccPools;

        // All Combined Type Generation Functions
        // Type Generation Functions
        std::map<std::string, GPUPrimGroupGen>          primGroupGenerators;
        std::map<std::string, GPUAccelGroupGen>         accelGroupGenerators;
        std::map<std::string, GPUMatGroupGen>           matGroupGenerators;

        std::map<std::string, GPUAccelBatchGen>         accelBatchGenerators;
        std::map<std::string, GPUMatBatchGen>           matBatchGenerators;

        std::map<std::string, GPUBaseAccelGen>          baseAccelGenerators;

        // Generated Types (Called by GPU Scene)
        // These hold ownership of classes (thus these will force destruction)
        // Primitives
        std::map<std::string, GPUPrimGPtr>          primGroups;
        // Accelerators (Batch and Group)
        std::map<std::string, GPUAccelGPtr>         accelGroups;
        std::map<std::string, GPUAccelBPtr>         accelBatches;
        // Materials (Batch and Group)
        std::map<NameIdPair, GPUMatGPtr>            matGroups;
        std::map<NameIdPair, GPUMatBPtr>            matBatches;
        // Base Accelerator (Unique Ptr)
        GPUBaseAccelPtr                             baseAccelerator;
        // Tracer (Unique Ptr)
        GPUTracerPtr                                tracerPtr;

        // Generated Batch Mappings
        AcceleratorBatchMappings                    accelBatchMap;
        MaterialBatchMappings                       matBatchMap;

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
        SceneError                  GeneratePrimitiveGroup(GPUPrimitiveGroupI*&,
                                                           const std::string& primitiveType) override;
        // Accelerator
        SceneError                  GenerateAcceleratorGroup(GPUAcceleratorGroupI*&,
                                                             const GPUPrimitiveGroupI&,
                                                             const TransformStruct* t,
                                                             const std::string& accelType) override;
        SceneError                  GenerateAcceleratorBatch(GPUAcceleratorBatchI*&,
                                                             const GPUAcceleratorGroupI&,
                                                             const GPUPrimitiveGroupI&,
                                                             uint32_t keyBatchId) override;
        // Material
        SceneError                  GenerateMaterialGroup(GPUMaterialGroupI*&,
                                                          const std::string& materialType,
                                                          const int gpuId) override;
        SceneError                  GenerateMaterialBatch(GPUMaterialBatchI*&,
                                                          const GPUMaterialGroupI&,
                                                          const GPUPrimitiveGroupI&,
                                                          uint32_t keyBatchId) override;

        // Base Accelerator should be fetched after all the stuff is generated
        SceneError                  GenerateBaseAccelerator(GPUBaseAcceleratorI*&,
                                                            const std::string& accelType) override;
        // Finally get the tracer logic
        // Tracer logic will be constructed with respect to
        // Constructed batches
        DLLError                    GenerateBaseLogic(TracerBaseLogicI*&,
                                                      // Args
                                                      const TracerParameters& opts,
                                                      const Vector2i maxMats,
                                                      const Vector2i maxAccels,
                                                      const HitKey baseBoundMatKey,
                                                      // DLL Location
                                                      const std::string& libName,
                                                      const SharedLibArgs& mangledName) override;

        PrimitiveGroupList          GetPrimitiveGroups() const override;
        AcceleratorGroupList        GetAcceleratorGroups() const override;
        AcceleratorBatchMappings    GetAcceleratorBatches() const override;
        MaterialGroupList           GetMaterialGroups() const override;
        MaterialBatchMappings       GetMaterialBatches() const override;

        GPUBaseAcceleratorI*        GetBaseAccelerator() const override;

        // Resetting all generated Groups and Batches
        void                        ClearAll() override;

        // Inclusion Functionality
        // Additionally includes the materials, primitives etc. from other libraries
        DLLError                    IncludeBaseAcceleratorsFromDLL(const std::string& regex,
                                                                   const std::string& libName,
                                                                   const SharedLibArgs& mangledName) override;
         DLLError                   IncludeAcceleratorsFromDLL(const std::string& regex,
                                                               const std::string& libName,
                                                               const SharedLibArgs& mangledName) override;
         DLLError                   IncludeMaterialsFromDLL(const std::string& regex,
                                                            const std::string& libName,
                                                            const SharedLibArgs& mangledName) override;
         DLLError                   IncludePrimitivesFromDLL(const std::string& regex,
                                                             const std::string& libName,
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