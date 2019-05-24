#pragma once
/**

Base implementation of Logic Generator

Provides storage of types (via across boundary capable unique ptrs)
and it adds default accelerators and primitives as default types.

*/

#include <map>

#include "RayLib/Types.h"
#include "TracerLogicGeneratorI.h"
#include "DefaultTypeGenerators.h"

using NameIdPair = std::pair<std::string, int>;

class TracerLogicGenerator : public TracerLogicGeneratorI
{
    private:
    protected:
        // Type Generation Functions
        std::map<std::string, GPUPrimGroupGen>          primGroupGenerators;
        std::map<std::string, GPUAccelGroupGen>         accelGroupGenerators;
        std::map<std::string, GPUMatGroupGen>           matGroupGenerators;

        std::map<std::string, GPUAccelBatchGen>         accelBatchGenerators;
        std::map<std::string, GPUMatBatchGen>           matBatchGenerators;

        std::map<std::string, GPUBaseAccelGen>          baseAccelGenerators;

        // Generated Types
        // These hold ownership of classes (thus these will force destruction)
        // Primitives
        std::map<std::string, GPUPrimGPtr>              primGroups;
        // Accelerators (Batch and Group)
        std::map<std::string, GPUAccelGPtr>             accelGroups;
        std::map<std::string, GPUAccelBPtr>             accelBatches;
        // Materials (Batch and Group)
        std::map<NameIdPair, GPUMatGPtr>                matGroups;
        std::map<NameIdPair, GPUMatBPtr>                matBatches;

        GPUBaseAccelPtr                                 baseAccelerator;

        // Tracer Related
        GPUTracerGen                                    tracerGenerator;
        GPUTracerPtr                                    tracerLogic;

        // Generated Batch Mappings
        AcceleratorBatchMappings                        accelBatchMap;
        MaterialBatchMappings                           matBatchMap;

        // Helper Func
        uint32_t                                        CalculateHitStruct();

    public:
        // Constructor & Destructor
                                    TracerLogicGenerator(GPUTracerGen, GPUTracerPtr);
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
        SceneError                  GenerateBaseLogic(TracerBaseLogicI*&,
                                                      const TracerParameters& opts,
                                                      const Vector2i maxMats,
                                                      const Vector2i maxAccels,
                                                      const HitKey baseBoundMatKey) override;

        PrimitiveGroupList          GetPrimitiveGroups() const override;
        AcceleratorGroupList        GetAcceleratorGroups() const override;
        AcceleratorBatchMappings    GetAcceleratorBatches() const override;
        MaterialGroupList           GetMaterialGroups() const override;
        MaterialBatchMappings       GetMaterialBatches() const override;

        GPUBaseAcceleratorI*        GetBaseAccelerator() const override;

        // Resetting all generated Groups and Batches
        void                        ClearAll() override;

        // Inclusion Functionality
        // Additionally includes the materials from these libraries
        // No exclusion functionality provided just add what you need
        SceneError                  IncludeAcceleratorsFromDLL(const SharedLib&,
                                                               const std::string& mangledName = "\0") const override;
        SceneError                  IncludeMaterialsFromDLL(const SharedLib&,
                                                            const std::string& mangledName = "\0") const override;
        SceneError                  IncludePrimitivesFromDLL(const SharedLib&,
                                                             const std::string& mangledName = "\0") const override;
};
