#pragma once
/**

Base implementation of Logic Generator

Provides storage of types (via across boundary capable unique ptrs)
and it adds default accelerators and primitives as default types.

*/

#include <map>

#include "RayLib/Types.h"
#include "RayLib/SharedLib.h"

#include "TracerLogicGeneratorI.h"
#include "DefaultTypeGenerators.h"

using NameIdPair = std::pair<std::string, int>;

class TracerLogicGenerator : public TracerLogicGeneratorI
{
    private:
    protected:
        // Type Generation Functions
        std::map<std::string, GPUPrimGroupGen>      primGroupGenerators;
        std::map<std::string, GPUAccelGroupGen>     accelGroupGenerators;
        std::map<std::string, GPUMatGroupGen>       matGroupGenerators;

        std::map<std::string, GPUAccelBatchGen>     accelBatchGenerators;
        std::map<std::string, GPUMatBatchGen>       matBatchGenerators;

        std::map<std::string, GPUBaseAccelGen>      baseAccelGenerators;

        // Generated Types
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

        // Shared Libraries That are Loaded
        std::map<std::string, SharedLib>            openedLibs;

        // Helper Funcs
        uint32_t                                    CalculateHitStruct();
        bool                                        FindSharedLib(SharedLib* libOut, 
                                                                  const std::string& libName) const;

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

        // Exclusion functionality
        DLLError                    UnloadLibrary(std::string & libName) override;
        DLLError                    StripGenerators(std::string & regex) override;
};
