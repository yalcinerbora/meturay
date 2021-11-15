#include "OptixSystem.h"
#include <optix_function_table_definition.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "CudaSystem.h"
#include "OptixCheck.h"

#include "RayLib/FileSystemUtility.h"

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

    std::ifstream file(Utility::MergeFileFolder(Utility::CurrentExecPath(), fileName));
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
        if(gpu.Tier() == CudaGPU::GPU_KEPLER)
            continue;

        CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
        OptixDeviceContext context;
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &OptiXSystem::OptixLOG;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(0, &options, &context));

        //optixDevices.emplace_back(std::make_pair(gpu, context));
        optixDevices.emplace_back(gpu, context);
    }
}

OptiXSystem::~OptiXSystem()
{
    for(const auto& [gpu, context] : optixDevices)
    {
        OPTIX_CHECK(optixDeviceContextDestroy(context));
    }
}

const std::vector<OptiXSystem::OptixDevice>& OptiXSystem::OptixCapableDevices() const
{
    return optixDevices;
}