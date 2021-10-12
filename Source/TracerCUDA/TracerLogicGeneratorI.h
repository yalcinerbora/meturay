#pragma once

#include "RayLib/SceneStructs.h"
#include "RayLib/TracerStructs.h"
#include <string>

struct SceneError;
struct DLLError;
struct SharedLibArgs;

class SharedLib;
class CudaGPU;

struct TracerParameters;
class GPUSceneI;
class CudaSystem;

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
        virtual SceneError      GeneratePrimitiveGroup(GPUPrimGPtr&,
                                                       const std::string& primitiveType) = 0;
        // Accelerator
        virtual SceneError      GenerateAcceleratorGroup(GPUAccelGPtr&,
                                                         const GPUPrimitiveGroupI&,
                                                         const std::string& accelType) = 0;
        // Material
        virtual SceneError      GenerateMaterialGroup(GPUMatGPtr&,
                                                      const CudaGPU&,
                                                      const std::string& materialType) = 0;
        // Base Accelerator should be fetched after all the stuff is generated
        virtual SceneError      GenerateBaseAccelerator(GPUBaseAccelPtr&,
                                                        const std::string& accelType) = 0;
        // Medium
        virtual SceneError      GenerateMediumGroup(CPUMediumGPtr&,
                                                    const std::string& mediumType) = 0;
        // Transform
        virtual SceneError      GenerateTransformGroup(CPUTransformGPtr&,
                                                       const std::string& transformType) = 0;
        // Camera
        virtual SceneError      GenerateCameraGroup(CPUCameraGPtr&,
                                                    const std::string& cameraType) = 0;

        // Light
        virtual SceneError      GenerateLightGroup(CPULightGPtr&,
                                                   const CudaGPU&,
                                                   const GPUPrimitiveGroupI*,
                                                   const std::string& lightType) = 0;

        // Tracer Logic
        virtual SceneError      GenerateTracer(GPUTracerPtr&,
                                               const CudaSystem&,
                                               const GPUSceneI&,
                                               const TracerParameters&,
                                               const std::string& tracerType) = 0;

        //// Inclusion Functionality
        //// Additionally includes the materials from these libraries
        //// No exclusion functionality provided just add what you need
        //virtual DLLError    IncludeBaseAcceleratorsFromDLL(const std::string& libName,
        //                                                   const std::string& regex,
        //                                                   const SharedLibArgs& mangledName) = 0;
        //virtual DLLError    IncludeAcceleratorsFromDLL(const std::string& libName,
        //                                               const std::string& regex,
        //                                               const SharedLibArgs& mangledName) = 0;
        //virtual DLLError    IncludeMaterialsFromDLL(const std::string& libName,
        //                                            const std::string& regex,
        //                                            const SharedLibArgs& mangledName) = 0;
        //virtual DLLError    IncludePrimitivesFromDLL(const std::string& libName,
        //                                             const std::string& regex,
        //                                             const SharedLibArgs& mangledName) = 0;
        //virtual DLLError    IncludeTracersFromDLL(const std::string& libName,
        //                                          const std::string& regex,
        //                                          const SharedLibArgs& mangledName) = 0;
        //// Exclusion functionality
        //// Unload A Library
        //virtual DLLError    UnloadLibrary(std::string& libName) = 0;
        //// Resetting all generated Groups and Batches
        //virtual void            UnloadAll() = 0;
        //// Functionality Stripping
        //virtual DLLError    StripGenerators(std::string& regex) = 0;
};