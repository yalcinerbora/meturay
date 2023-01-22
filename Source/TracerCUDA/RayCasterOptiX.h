#pragma once


#include "RayCaster.h"
#include "OptixSystem.h"
#include "GPUOptiXPTX.cuh"
#include "RayMemory.h"

class GPUSceneI;
class GPUAcceleratorI;

class RayCasterOptiX : public RayCaster
{
    public:
        static constexpr const char* MODULE_BASE_NAME       = "OptiXShaders/GPUOptiXPTX.optixir";
        static constexpr const char* RAYGEN_FUNC_NAME       = "__raygen__OptiX";
        static constexpr const char* MISS_FUNC_NAME         = "__miss__OptiX";
        static constexpr const char* CHIT_FUNC_PREFIX       = "__closesthit__";
        static constexpr const char* AHIT_FUNC_PREFIX       = "__anyhit__";
        static constexpr const char* INTERSECT_FUNC_PREFIX  = "__intersection__";

        static constexpr int CH_INDEX   = 0;
        static constexpr int AH_INDEX   = 1;
        static constexpr int INTS_INDEX = 2;

        using HitFunctionNames          = std::tuple<std::string, std::string, std::string>;
        using ProgramGroups             = std::vector<OptixProgramGroup>;

        struct OptixGPUData
        {
            OptixTraversableHandle      baseAccelerator;

            OptixPipeline               pipeline;
            OptixModule                 mdl;
            ProgramGroups               programGroups;
            // Host copy of the Params
            OpitXBaseAccelParams        hOptixLaunchParams;
            // Device copy of the params
            OpitXBaseAccelParams*       dOptixLaunchParams;
            DeviceLocalMemory           paramsMemory;
            // SBT
            OptixShaderBindingTable     sbt;
            DeviceLocalMemory           sbtMemory;
        };

    private:
        // OptiX Related
        OptiXSystem                 optixSystem;
        std::vector<OptixGPUData>   optixGPUData;
        // GPU Memory
        const GPUTransformI**       dGlobalTransformArray;

        // Functions
        TracerError                 CreateProgramGroups(const std::string& rgFuncName,
                                                        const std::string& missFuncName,
                                                        const std::vector<HitFunctionNames>&);
        TracerError                 CreateModules(const OptixModuleCompileOptions& mOpts,
                                                  const OptixPipelineCompileOptions& pOpts,
                                                  const std::string& baseFileName);
        TracerError                 CreatePipelines(const OptixPipelineCompileOptions& pOpts,
                                                    const OptixPipelineLinkOptions& lOpts);
        TracerError                 CreateSBTs(const std::vector<Record<void,void>>& recordPointers,
                                               const std::vector<uint32_t>& programGroupIds);
        TracerError                 AllocateParams();

    protected:
    public:
        // Constructors & Destructor
                                    RayCasterOptiX(const GPUSceneI& gpuScene,
                                                   const CudaSystem& system);
                                    RayCasterOptiX(const RayCasterOptiX&) = delete;
        RayCasterOptiX&             operator=(const RayCasterOptiX&) = delete;
                                    ~RayCasterOptiX();

        // Interface
        TracerError                 ConstructAccelerators(const GPUTransformI** dTransforms,
                                                          uint32_t identityTransformIndex) override;
        void                        HitRays() override;

        // Memory Usage
        size_t                      UsedGPUMemory() const override;
};