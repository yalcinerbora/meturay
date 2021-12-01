#pragma once


#include "RayCasterI.h"
#include "OptixSystem.h"
#include "GPUOptiXPTX.cuh"
#include "RayMemory.h"

class GPUSceneI;
class GPUAcceleratorI;

class RayCasterOptiX : public RayCasterI
{
    public:
        static constexpr const char* MODULE_BASE_NAME       = "OptiXShaders/GPUOptiXPTX.o.ptx";
        static constexpr const char* RAYGEN_FUNC_NAME       = "__raygen__OptiX";
        static constexpr const char* MISS_FUNC_NAME         = "__miss__OptiX";
        static constexpr const char* CHIT_FUNC_PREFIX       = "__closesthit__";
        static constexpr const char* AHIT_FUNC_PREFIX       = "__anyhit__";
        static constexpr const char* INTERSECT_FUNC_PREFIX  = "__intersection__";

        static constexpr int CH_INDEX = 0;
        static constexpr int AH_INDEX = 1;
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
        std::vector<OptixGPUData>       optixGPUData;
        // GPU Memory
        RayMemory                       rayMemory;
        const GPUTransformI**           dGlobalTransformArray;
        // Debug
        OptixTraversableHandle          gas;

        // Funcs
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
        // Mem Usage
        size_t                  UsedGPUMemory() const override;
};

inline uint32_t RayCasterOptiX::CurrentRayCount() const
{
    return currentRayCount;
}

inline void RayCasterOptiX::ResizeRayOut(uint32_t rayCount,
                                         HitKey baseBoundMatKey)
{
    currentRayCount = rayCount;
    return rayMemory.ResizeRayOut(rayCount, baseBoundMatKey);
}

inline RayGMem* RayCasterOptiX::RaysOut()
{
    return rayMemory.RaysOut();
}

inline void RayCasterOptiX::SwapRays()
{
    rayMemory.SwapRays();
}

inline void RayCasterOptiX::OverrideWorkBits(const Vector2i newWorkBits)
{
    maxWorkBits = newWorkBits;
}