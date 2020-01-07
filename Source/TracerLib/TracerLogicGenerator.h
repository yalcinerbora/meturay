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
using TracerLogicPoolPtr = SharedLibPtr<TracerLogicPoolI>;
using EstimatorPoolPtr = SharedLibPtr<EstimatorLogicPoolI>;

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
        std::map<PoolKey, TracerLogicPoolPtr>       loadedTracerPools;
        std::map<PoolKey, EstimatorPoolPtr>         loadedEstimatorPools;

        // All Combined Type Generation Functions
        // Type Generation Functions
        std::map<std::string, GPUPrimGroupGen>      primGroupGenerators;
        std::map<std::string, GPUAccelGroupGen>     accelGroupGenerators;
        std::map<std::string, GPUMatGroupGen>       matGroupGenerators;

        std::map<std::string, GPUAccelBatchGen>     accelBatchGenerators;
        std::map<std::string, GPUMatBatchGen>       matBatchGenerators;

        std::map<std::string, GPUBaseAccelGen>      baseAccelGenerators;

        std::map<std::string, GPUEstimatorGen>      estimatorGenerators;
        std::map<std::string, GPUTracerGen>         tracerGenerators;

        // Generated Types (Called by GPU Scene)
        // These hold ownership of classes (thus these will force destruction)
        // Primitives
        std::map<std::string, GPUPrimGPtr>          primGroups;
        // Accelerators (Batch and Group)
        std::map<std::string, GPUAccelGPtr>         accelGroups;
        std::map<std::string, GPUAccelBPtr>         accelBatches;
        // Materials (Batch and Group)
        std::map<NameGPUPair, GPUMatGPtr>           matGroups;
        std::map<NameGPUPair, GPUMatBPtr>           matBatches;
        // Base Accelerator (Unique Ptr)
        GPUBaseAccelPtr                             baseAccelerator;
        // Tracer (Unique Ptr)
        GPUTracerPtr                                tracerPtr;
        // Estimator (Unique Ptr)
        GPUEstimatorPtr                             estimatorPtr;

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
                                                             uint32_t keyBatchId,
                                                             const std::string& batchType) override;
        // Material
        SceneError                  GenerateMaterialGroup(GPUMaterialGroupI*&,
                                                          const CudaGPU&,
                                                          const GPUEventEstimatorI&,
                                                          const std::string& materialType) override;
        SceneError                  GenerateMaterialBatch(GPUMaterialBatchI*&,
                                                          const GPUMaterialGroupI&,
                                                          const GPUPrimitiveGroupI&,
                                                          uint32_t keyBatchId,
                                                          const std::string& batchType) override;
        // Base Accelerator should be fetched after all the stuff is generated
        SceneError                  GenerateBaseAccelerator(GPUBaseAcceleratorI*&,
                                                            const std::string& accelType) override;
        // EventEstimator
        SceneError                  GenerateEventEstimaor(GPUEventEstimatorI*&,
                                                          const std::string& estType) override;
        // Tracer Logic
        SceneError                  GenerateTracerLogic(TracerBaseLogicI*&,
                                                        // Args
                                                        const TracerParameters& opts,
                                                        const Vector2i maxMats,
                                                        const Vector2i maxAccels,
                                                        const HitKey baseBoundMatKey,
                                                        // Type
                                                        const std::string& tracerType) override;

        PrimitiveGroupList          GetPrimitiveGroups() const override;
        AcceleratorGroupList        GetAcceleratorGroups() const override;
        AcceleratorBatchMappings    GetAcceleratorBatches() const override;
        MaterialGroupList           GetMaterialGroups() const override;
        MaterialBatchMappings       GetMaterialBatches() const override;

        GPUBaseAcceleratorI*        GetBaseAccelerator() const override;
        GPUEventEstimatorI*         GetEventEstimator() const  override;
        TracerBaseLogicI*           GetTracerLogic() const override;

        // Resetting all generated Groups and Batches
        void                        ClearAll() override;

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