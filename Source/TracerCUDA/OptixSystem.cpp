#include "OptixSystem.h"
#include <optix_function_table_definition.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <optix_stack_size.h>

#include "CudaSystem.h"
#include "OptixCheck.h"

#include <fstream>

void OptiXSystem::OptixLOG(unsigned int level,
                           const char* tag,
                           const char* message,
                           void*)
{
    try
    {
        static auto logger = spdlog::basic_logger_mt("OptixLog", "optix_log");
        switch(level)
        {
            case 1: logger->error("[FATAL]{:s}, {:s}\n", tag, message);
            case 2: logger->error("{:s}, {:s}\n", tag, message);
            case 3: logger->warn("{:s}, {:s}\n", tag, message);
            case 4: logger->info("{:s}, {:s}\n", tag, message);
        }
    }
    catch(const spdlog::spdlog_ex& ex)
    {
        METU_ERROR_LOG("Log init failed: {:S}", ex.what());
    }
}

TracerError OptiXSystem::LoadPTXFile(std::string& ptxSource,
                                     const CudaGPU& gpu,
                                     const std::string& baseName)
{
    std::string fileName = baseName;
    std::string ccName = "CC_" + gpu.CC();

    auto loc = fileName.find_last_of(".o.ptx");
    fileName.insert(loc, ccName);

    std::ifstream file(fileName);
    if(!file.is_open())
        return TracerError(TracerError::OPTIX_PTX_FILE_NOT_FOUND, fileName);

    std::stringstream buffer;
    buffer << file.rdbuf();
    ptxSource = buffer.str();
    return TracerError::OK;
}

OptiXSystem::OptiXSystem(const CudaSystem& system)
    : cudaSystem(system)
{
    OPTIX_CHECK(optixInit());

    for(const CudaGPU& gpu : cudaSystem.SystemGPUs())
    {
        CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
        deviceStates.emplace_back();
        CudaGPU::GPUTier tier = gpu.Tier();
        if(tier != CudaGPU::GPU_MAXWELL ||
           tier != CudaGPU::GPU_PASCAL ||
           tier != CudaGPU::GPU_TURING_VOLTA ||
           tier != CudaGPU::GPU_AMPERE)
            continue;

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &OptiXSystem::OptixLOG;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(0, &options, &deviceStates.back().context));
    }
}

OptiXSystem::~OptiXSystem()
{
    for(const auto ctx : deviceStates)
    {
        OPTIX_CHECK(optixPipelineDestroy(ctx.pipeline));
        OPTIX_CHECK(optixModuleDestroy(ctx.module));
        OPTIX_CHECK(optixDeviceContextDestroy(ctx.context));
    }
}

const std::vector<OptiXSystem::OptixDevice>& OptiXSystem::OptixCapableDevices() const
{
    return optixDevices;
}

TracerError OptiXSystem::OptixGenerateModules(const OptixModuleCompileOptions& mOpts,
                                              const OptixPipelineCompileOptions& pOpts,
                                              const std::string& baseFileName)
{
    TracerError err = TracerError::OK;

    int i = 0;
    for(const CudaGPU& gpu : cudaSystem.SystemGPUs())
    {
        std::string ptxSource;
        if((err = LoadPTXFile(ptxSource, gpu, baseFileName)) != TracerError::OK)
            return err;

        OPTIX_CHECK(optixModuleCreateFromPTX(deviceStates[i].context,
                                             &mOpts, &pOpts,
                                             ptxSource.c_str(),
                                             ptxSource.size(),
                                             nullptr,
                                             nullptr,
                                             &deviceStates[i].module));
        i++;
    }
    return TracerError::OK;
}

TracerError OptiXSystem::OptixGeneratePipelines(const OptixPipelineCompileOptions& pOpts,
                                                const OptixPipelineLinkOptions& lOpts,
                                                const std::vector<OptixProgramGroup>& programs)
{
    TracerError err = TracerError::OK;

    int i = 0;
    for(const CudaGPU& gpu : cudaSystem.SystemGPUs())
    {
        OPTIX_CHECK(optixPipelineCreate(deviceStates[i].context,
                                        &pOpts, &lOpts,
                                        programs.data(),
                                        static_cast<uint32_t>(programs.size()),
                                        nullptr, nullptr,
                                        &deviceStates[i].pipeline));

            // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
        // parameters to optixPipelineSetStackSize.
        OptixStackSizes stack_sizes = {};
        for(const auto& pg : programs)
            OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes));

        uint32_t dcStackSizeTraverse;
        uint32_t dcStackSizeState;
        uint32_t contStackSize;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
                                               2,   // max trace depth
                                               0, 0,
                                               &dcStackSizeTraverse,
                                               &dcStackSizeState,
                                               &contStackSize));

        const uint32_t maxTraversalDepth = 2;
        OPTIX_CHECK(optixPipelineSetStackSize(deviceStates[i].pipeline,
                                              dcStackSizeTraverse,
                                              dcStackSizeState,
                                              contStackSize,
                                              maxTraversalDepth));

        i++;
    }
    return TracerError::OK;
}

OptixDeviceContext OptiXSystem::OptixContext(const CudaGPU& gpu) const
{
    int i = 0;
    for(const CudaGPU& g : cudaSystem.SystemGPUs())
    {
        if(g.DeviceId() == gpu.DeviceId())
            return deviceStates[i].context;
        i++;
    }
    return nullptr;
}

OptixModule OptiXSystem::OptixModule(const CudaGPU& gpu) const
{
    int i = 0;
    for(const CudaGPU& g : cudaSystem.SystemGPUs())
    {
        if(g.DeviceId() == gpu.DeviceId())
            return deviceStates[i].module;
        i++;
    }
    return nullptr;
}