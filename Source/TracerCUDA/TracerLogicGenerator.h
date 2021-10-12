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

class TracerLogicGenerator : public TracerLogicGeneratorI
{
    private:
    protected:
        // All Combined Type Generation Functions
        // Type Generation Functions
        std::map<std::string, GPUPrimGroupGen>      primGroupGenerators;
        std::map<std::string, GPUAccelGroupGen>     accelGroupGenerators;
        std::map<std::string, GPUMatGroupGen>       matGroupGenerators;
        std::map<std::string, GPUBaseAccelGen>      baseAccelGenerators;
        std::map<std::string, GPUTracerGen>         tracerGenerators;
        std::map<std::string, CPUTransformGen>      transGroupGenerators;
        std::map<std::string, CPUMediumGen>         medGroupGenerators;
        std::map<std::string, CPULightGroupGen>     lightGroupGenerators;
        std::map<std::string, CPUCameraGen>         camGroupGenerators;

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
                                                             const std::string& accelType) override;
        // Material
        SceneError                  GenerateMaterialGroup(GPUMatGPtr&,
                                                          const CudaGPU&,
                                                          const std::string& materialType) override;
        // Base Accelerator should be fetched after all the stuff is generated
        SceneError                  GenerateBaseAccelerator(GPUBaseAccelPtr&,
                                                            const std::string& accelType) override;
        // Medium
        SceneError                  GenerateMediumGroup(CPUMediumGPtr&,
                                                        const std::string& mediumType) override;
        // Transform
        SceneError                  GenerateTransformGroup(CPUTransformGPtr&,
                                                           const std::string& transformType) override;
        // Camera
        SceneError                  GenerateCameraGroup(CPUCameraGPtr&,
                                                        const std::string& cameraType) override;
        // Light
        SceneError                  GenerateLightGroup(CPULightGPtr&,
                                                       const CudaGPU&,
                                                       const GPUPrimitiveGroupI*,
                                                       const std::string& lightType) override;
        // Tracer Logic
        SceneError                  GenerateTracer(GPUTracerPtr&,
                                                   const CudaSystem&,
                                                   const GPUSceneI&,
                                                   const TracerParameters&,
                                                   const std::string& tracerType) override;
};