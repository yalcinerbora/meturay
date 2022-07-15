#pragma once

#include "RayLib/TracerSystemI.h"
#include "RayLib/SurfaceLoaderGenerator.h"

#include "CudaSystem.h"
#include "ScenePartitionerI.h"
#include "TracerLogicGeneratorI.h"

class TracerSystemCUDA final : public TracerSystemI
{
    private:
        std::unique_ptr<CudaSystem>                 cudaSystem;
        std::unique_ptr<TracerLogicGeneratorI>      logicGenerator;

        std::unique_ptr<ScenePartitionerI>          scenePartitioner;
        std::unique_ptr<SurfaceLoaderGenerator>     surfaceLoaders;
        std::unique_ptr<GPUSceneI>                  gpuScene;
        GPUReconFilterPtr                           reconFilter;

    protected:
    public:
        // Constructors & Destructor
                                TracerSystemCUDA();
                                TracerSystemCUDA(const TracerSystemCUDA&) = delete;
                                TracerSystemCUDA(TracerSystemCUDA&&) = delete;
        TracerSystemCUDA&       operator=(const TracerSystemCUDA&) = delete;
        TracerSystemCUDA&       operator=(TracerSystemCUDA&&) = delete;

        TracerError             Initialize(const std::vector<SurfaceLoaderSharedLib>&,
                                           ScenePartitionerType) override;
        void                    ClearScene() override;
        void                    GenerateScene(GPUSceneI*&,
                                              const std::u8string& scenePath,
                                              SceneLoadFlags = {}) override;
        TracerError             GenerateTracer(GPUTracerPtr&,
                                               const TracerParameters&,
                                               const Options&,
                                               const std::string& tracerType) override;
        TracerError             GenerateReconFilter(GPUReconFilterI*&,
                                                    const Options&) override;
};