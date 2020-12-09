#include "TracerSystemCUDA.h"
#include "TracerLogicGenerator.h"

#include "RayLib/SurfaceLoaderGenerator.h"
#include "RayLib/GPUTracerI.h"
#include "RayLib/TracerOptions.h"
#include "RayLib/Log.h"

#include "ScenePartitioner.h"
#include "GPUSceneJson.h"

TracerError TracerSystemCUDA::Initialize(const std::vector<SurfaceLoaderSharedLib>& sLoaderArgs,
                                         ScenePartitionerType partitionerType)
{
    // Load Surface Loader DLLs
    DLLError dllE = DLLError::OK;
    surfaceLoaders = std::make_unique<SurfaceLoaderGenerator>();
    for(const SurfaceLoaderSharedLib& sl : sLoaderArgs)
    {
        dllE = surfaceLoaders->IncludeLoadersFromDLL(sl.libName,
                                                     sl.regex,
                                                     sl.mangledName);
        if(dllE != DLLError::OK)
        {
            METU_ERROR_LOG("%s", static_cast<std::string>(dllE).c_str());
            return TracerError::UNABLE_TO_INITIALIZE;
        }
    }

    cudaSystem = std::make_unique<CudaSystem>();
    CudaError cudaE = cudaSystem->Initialize();
    if(cudaE != CudaError::OK)
    {
        METU_ERROR_LOG("%s", static_cast<std::string>(cudaE).c_str());
        return TracerError::UNABLE_TO_INITIALIZE;
    }
    switch(partitionerType)
    {
        case ScenePartitionerType::SINGLE_GPU:
            scenePartitioner = std::make_unique<SingleGPUScenePartitioner>(*cudaSystem);
            break;
        case ScenePartitionerType::MULTI_GPU_MATERIAL:
        default:
            return TracerError::UNKNOWN_SCENE_PARTITIONER_TYPE;
            break;
    }
    return TracerError::OK;
}

void TracerSystemCUDA::GenerateScene(GPUSceneI*& newScene,
                                     const std::u8string& scenePath)
{
    // Clear Logic List by constructing new logic generator
    logicGenerator = std::make_unique<TracerLogicGenerator>();
    // Override old scene with new scene
    gpuScene = std::make_unique<GPUSceneJson>(scenePath,
                                              *scenePartitioner,
                                              *logicGenerator,
                                              *surfaceLoaders,
                                              *cudaSystem);
    newScene = gpuScene.get();
}

TracerError TracerSystemCUDA::GenerateTracer(GPUTracerPtr& tracer,
                                             const TracerParameters& params,
                                             const TracerOptions& opts,
                                             const std::string& tracerType)
{
    SceneError scnE = logicGenerator->GenerateTracer(tracer,
                                                     *cudaSystem,
                                                     *gpuScene,
                                                     params,
                                                     tracerType);
    if(scnE != SceneError::OK)
    {
        return TracerError::NO_LOGIC_FOR_TRACER;
    }        
    TracerError trcE = tracer->SetOptions(opts);
    if(trcE != TracerError::OK)
        return trcE;
    return TracerError::OK;
}