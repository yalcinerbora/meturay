#pragma once


#include "RayCasterI.h"
#include "OptixSystem.h"
#include "GPUAcceleratorOptiXKC.cuh"
#include "GPUOptixPTX.cuh"

class GPUSceneI;
class GPUAcceleratorI;

class RayCasterOptiX : public RayCasterI
{
    public:
        static constexpr const char*    MODULE_BASE_NAME = "OptiXShaders/GPUAcceleratorOptiXKC.o.ptx";
        static constexpr const char*    RAYGEN_FUNC_NAME = "KCRayGenOptix";

        using HitFunctionNames          = std::tuple<std::string, std::string, std::string>;
        using ProgramGroups             = std::vector<OptixProgramGroup>;
        //struct ProgramGroups
        //{
        //    OptixProgramGroup              raygenProgram;
        //    std::vector<OptixProgramGroup> acceleratorHitPrograms;
        //};

        struct OptixGPUData
        {
            DeviceLocalMemory           lpMemory;
            OpitXBaseAccelParams*       dOptixLaunchParams;
            OptixPipeline               pipeline;
            OptixModule                 mdl;
            ProgramGroups               programGroups;
            OptixShaderBindingTable     sbt;
        };

    private:
        Vector2i                        maxAccelBits;
        Vector2i                        maxWorkBits;
        // Combined hit struct size
        const uint32_t                  maxHitSize;
        const uint32_t                  boundaryTransformIndex;
        // Cuda System for GPU Kernel Launces
        const CudaSystem&               cudaSystem;
        // Accelerators
        GPUBaseAcceleratorI&            baseAccelerator;
        const AcceleratorBatchMap&      accelBatches;
        // Misc
        uint32_t                        currentRayCount;
        // OptiX Related
        OptiXSystem                     optixSystem;
        // CPU Memory
        // GPU Memory
        std::vector<OptixGPUData>       optixGPUData;

        DeviceMemory                    globalRayInMemory;
        DeviceMemory                    globalRayOutMemory;
        //...

        // Funcs
        TracerError                 CreateProgramGroups(const std::string& rgFuncName,
                                                        const std::vector<HitFunctionNames>&);
        TracerError                 CreateModules(const OptixModuleCompileOptions& mOpts,
                                                  const OptixPipelineCompileOptions& pOpts,
                                                  const std::string& baseFileName);
        TracerError                 CreatePipelines(const OptixPipelineCompileOptions& pOpts,
                                                    const OptixPipelineLinkOptions& lOpts);

    protected:
    public:
        // Constructors & Destructor
                                    RayCasterOptiX(const GPUSceneI& gpuScene,
                                                   const CudaSystem& system);
                                    RayCasterOptiX(const RayCasterOptiX&) = delete;
        RayCasterOptiX&             operator=(const RayCasterOptiX&) = delete;
                                    ~RayCasterOptiX() = default;

        // Interface
        TracerError                 ConstructAccelerators(const GPUTransformI** dTransforms,
                                                          uint32_t identityTransformIndex) override;
        RayPartitions<uint32_t>     HitAndPartitionRays() override;
        void                        WorkRays(const WorkBatchMap& workMap,
                                             const RayPartitionsMulti<uint32_t>& outPortions,
                                             const RayPartitions<uint32_t>& inPartitions,
                                             RNGMemory& rngMemory,
                                             uint32_t totalRayOut,
                                             HitKey baseBoundMatKey) override;

        // Ray Related
        uint32_t                CurrentRayCount() const override;
        void                    ResizeRayOut(uint32_t rayCount,
                                             HitKey baseBoundMatKey) override;
        RayGMem*                RaysOut() override;
        void                    SwapRays() override;
        // Work Related
        void                    OverrideWorkBits(const Vector2i newWorkBits) override;
};


inline uint32_t RayCasterOptiX::CurrentRayCount() const
{
    return currentRayCount;
}

inline void RayCasterOptiX::OverrideWorkBits(const Vector2i newWorkBits)
{
    maxWorkBits = newWorkBits;
}