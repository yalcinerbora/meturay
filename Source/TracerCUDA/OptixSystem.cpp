#include "OptixSystem.h"
#include <optix_function_table_definition.h>

#include "CudaSystem.h"
#include "OptixCheck.h"

#include "RayLib/FileSystemUtility.h"

#include <fstream>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

void OptiXSystem::OptixLOG(unsigned int level,
                           const char* tag,
                           const char* message,
                           void*)
{
    auto logger = spdlog::get(OPTIX_LOGGER_NAME);
    switch(level)
    {
        case 1: logger->error("[FATAL]{:s}, {:s}", tag, message); break;
        case 2: logger->error("{:s}, {:s}", tag, message); break;
        case 3: logger->warn("{:s}, {:s}", tag, message); break;
        case 4: logger->info("{:s}, {:s}", tag, message); break;
    }
    logger->flush();
}

TracerError OptiXSystem::LoadPTXFile(std::vector<Byte>& ptxSource,
                                     const CudaGPU& gpu,
                                     const std::string& baseName)
{
    std::string fileName = baseName;

    // TODO: Change this later to better implementation
    #ifdef MRAY_OPTIX_USE_NATIVE_CC
        std::string ccName = ".CC_native";
    #else
        std::string ccName = ".CC_" + gpu.CC();
    #endif

    auto loc = fileName.rfind(".optixir");
    fileName.insert(loc, ccName);

    std::string fullFileName = Utility::MergeFileFolder(Utility::CurrentExecPath(),
                                                        fileName);
    std::ifstream file(fullFileName, std::ios::binary | std::ios::ate);
    if(!file.is_open())
        return TracerError(TracerError::OPTIX_PTX_FILE_NOT_FOUND, fileName);

    std::ifstream::pos_type size = file.tellg();
    ptxSource.resize(size);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(ptxSource.data()), size);
    return TracerError::OK;
}

OptiXSystem::OptiXSystem(const CudaSystem& system)
    : cudaSystem(system)
{
    OPTIX_CHECK(optixInit());

    try
    {
        auto logger = spdlog::basic_logger_mt(OPTIX_LOGGER_NAME,
                                              OPTIX_LOGGER_FILE_NAME,
                                              true);
    }
    catch(const spdlog::spdlog_ex& ex)
    {
        METU_ERROR_LOG("Log init failed: {:S}", ex.what());
    }

    for(const CudaGPU& gpu : cudaSystem.SystemGPUs())
    {
        if(gpu.Tier() == CudaGPU::GPU_KEPLER)
            continue;

        CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
        OptixDeviceContext context;
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &OptiXSystem::OptixLOG;
        options.logCallbackLevel = 4;
        if constexpr(METU_DEBUG_BOOL)
            options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

        OPTIX_CHECK(optixDeviceContextCreate(0, &options, &context));

        // Disable caching for debugging purposes
        // TODO: change this
        optixDeviceContextSetCacheEnabled(context, 0);

        optixDevices.emplace_back(gpu, context);
    }
}

OptiXSystem::~OptiXSystem()
{
    for(const auto& [gpu, context] : optixDevices)
    {
        OPTIX_CHECK(optixDeviceContextDestroy(context));
    }
    spdlog::drop(OPTIX_LOGGER_NAME);
}

const std::vector<OptiXSystem::OptixDevice>& OptiXSystem::OptixCapableDevices() const
{
    return optixDevices;
}