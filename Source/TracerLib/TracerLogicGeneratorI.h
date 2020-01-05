#pragma once

#include "RayLib/SceneStructs.h"
#include <string>

struct SceneError;
struct DLLError;
struct SharedLibArgs;

class SharedLib;
class CudaGPU;

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
// Event Estimator
class GPUEventEstimatorI;

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
                                                      const CudaGPU&,
                                                      const GPUEventEstimatorI&) = 0;
        virtual SceneError      GenerateMaterialBatch(GPUMaterialBatchI*&,
                                                      const GPUMaterialGroupI&,
                                                      const GPUPrimitiveGroupI&,
                                                      uint32_t keyBatchId) = 0;
        // Base Accelerator should be fetched after all the stuff is generated
        virtual SceneError      GenerateBaseAccelerator(GPUBaseAcceleratorI*&,
                                                        const std::string& accelType) = 0;
        // EventEstimator
        virtual SceneError      GenerateEventEstimaor(GPUEventEstimatorI*&, 
                                                      const std::string& estType) = 0;
        // Tracer Logic
        virtual SceneError      GenerateTracerLogic(TracerBaseLogicI*&,
                                                    // Args
                                                    const TracerParameters& opts,
                                                    const Vector2i maxMats,
                                                    const Vector2i maxAccels,
                                                    const HitKey baseBoundMatKey,
                                                    // Type
                                                    const std::string& tracerType) = 0;

        // Get all generated stuff on a vector
        virtual PrimitiveGroupList          GetPrimitiveGroups() const = 0;
        virtual AcceleratorGroupList        GetAcceleratorGroups() const = 0;
        virtual AcceleratorBatchMappings    GetAcceleratorBatches() const = 0;
        virtual MaterialGroupList           GetMaterialGroups() const = 0;
        virtual MaterialBatchMappings       GetMaterialBatches() const = 0;

        virtual GPUBaseAcceleratorI*        GetBaseAccelerator() const = 0;
        virtual GPUEventEstimatorI*         GetEventEstimator() const = 0;
        virtual TracerBaseLogicI*           GetTracerLogic() const = 0;

        // Resetting all generated Groups and Batches
        virtual void                        ClearAll() = 0;

        // Inclusion Functionality
        // Additionally includes the materials from these libraries
        // No exclusion functionality provided just add what you need
        virtual DLLError    IncludeBaseAcceleratorsFromDLL(const std::string& libName,
                                                           const std::string& regex,
                                                           const SharedLibArgs& mangledName) = 0;
        virtual DLLError    IncludeAcceleratorsFromDLL(const std::string& libName,
                                                       const std::string& regex,
                                                       const SharedLibArgs& mangledName) = 0;
        virtual DLLError    IncludeMaterialsFromDLL(const std::string& libName,
                                                    const std::string& regex,
                                                    const SharedLibArgs& mangledName) = 0;
        virtual DLLError    IncludePrimitivesFromDLL(const std::string& libName,
                                                     const std::string& regex,
                                                     const SharedLibArgs& mangledName) = 0;
        virtual DLLError    IncludeEstimatorsFromDLL(const std::string& libName,
                                                     const std::string& regex,
                                                     const SharedLibArgs& mangledName) = 0;
        virtual DLLError    IncludeTracersFromDLL(const std::string& libName,
                                                  const std::string& regex,
                                                  const SharedLibArgs& mangledName) = 0;
        //// Exclusion functionality
        //// Unload A Library
        //virtual DLLError    UnloadLibrary(std::string& libName) = 0;

        //// Functionality Stripping
        //virtual DLLError    StripGenerators(std::string& regex) = 0;
};