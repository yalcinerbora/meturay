#pragma once

#include "RayLib/SceneStructs.h"
#include <string>

struct SceneError;
struct SharedLibArgs;

class SharedLib;

// Execution Related Abstraction
class GPUBaseAcceleratorI;
class GPUAcceleratorBatchI;
class GPUMaterialBatchI;
// Data Related Abstraction
class GPUPrimitiveGroupI;
class GPUAcceleratorGroupI;
class GPUMaterialGroupI;
// Base Logic
class TracerBaseLogicI;
struct TracerParameters;

class TracerLogicGeneratorI
{
    public:
        virtual                 ~TracerLogicGeneratorI() = default;

        // Logic Generators
        // This is the heart of the type generation mechanism
        // of the DLL (A.K.A abstract factory)
        // It generates or returns (if already constructed) types
        // w.r.t. a type name and parent type if applicable
        // Pritimive
        virtual SceneError      GeneratePrimitiveGroup(GPUPrimitiveGroupI*&,
                                                       const std::string& primitiveType) = 0;
        // Accelerator
        virtual SceneError      GenerateAcceleratorGroup(GPUAcceleratorGroupI*&,
                                                         const GPUPrimitiveGroupI&,
                                                         const TransformStruct* t,
                                                         const std::string& accelType) = 0;
        virtual SceneError      GenerateAcceleratorBatch(GPUAcceleratorBatchI*&,
                                                         const GPUAcceleratorGroupI&,
                                                         const GPUPrimitiveGroupI&,
                                                         uint32_t keyBatchId) = 0;
        // Material
        virtual SceneError      GenerateMaterialGroup(GPUMaterialGroupI*&,
                                                      const std::string& materialType,
                                                      const int gpuId) = 0;
        virtual SceneError      GenerateMaterialBatch(GPUMaterialBatchI*&,
                                                      const GPUMaterialGroupI&,
                                                      const GPUPrimitiveGroupI&,
                                                      uint32_t keyBatchId) = 0;
        // Base Accelerator should be fetched after all the stuff is generated
        virtual SceneError      GenerateBaseAccelerator(GPUBaseAcceleratorI*&,
                                                        const std::string& accelType) = 0;

        // Finally get the tracer logic
        // Tracer logic will be constructed with respect to
        // Constructed batches
        virtual SceneError      GenerateBaseLogic(TracerBaseLogicI*&,
                                                  // Args
                                                  const TracerParameters& opts,
                                                  const Vector2i maxMats,
                                                  const Vector2i maxAccels,
                                                  const HitKey baseBoundMatKey,
                                                  // DLL Location
                                                  const std::string& libName,
                                                  const SharedLibArgs& mangledName) = 0;

        // Get all generated stuff on a vector
        virtual PrimitiveGroupList          GetPrimitiveGroups() const = 0;
        virtual AcceleratorGroupList        GetAcceleratorGroups() const = 0;
        virtual AcceleratorBatchMappings    GetAcceleratorBatches() const = 0;
        virtual MaterialGroupList           GetMaterialGroups() const = 0;
        virtual MaterialBatchMappings       GetMaterialBatches() const = 0;

        virtual GPUBaseAcceleratorI*        GetBaseAccelerator() const = 0;

        // Resetting all generated Groups and Batches
        virtual void                        ClearAll() = 0;

        // Inclusion Functionality
        // Additionally includes the materials from these libraries
        // No exclusion functionality provided just add what you need
        virtual SceneError      IncludeBaseAcceleratorsFromDLL(const std::string& libName,
                                                               const SharedLibArgs& mangledame) const = 0;
        virtual SceneError      IncludeAcceleratorsFromDLL(const std::string& libName,
                                                           const SharedLibArgs& mangledName) const = 0;
        virtual SceneError      IncludeMaterialsFromDLL(const std::string& libName,
                                                        const SharedLibArgs& mangledName) const = 0;
        virtual SceneError      IncludePrimitivesFromDLL(const std::string& libName,
                                                         const SharedLibArgs& mangledName) const = 0;

        // Logic Generator Inclusion
        virtual SceneError      AttachBaseLogicGenerator(const std::string& libName,
                                                         const SharedLibArgs& mangledName) const = 0;
};