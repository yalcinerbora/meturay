#include "GPUMetaSurfaceGenerator.h"
#include "RayLib/GPUSceneI.h"
#include "GPUWorkI.h"

TracerError GPUMetaSurfaceHandler::Initialize(const GPUSceneI& scene,
                                              const WorkBatchMap& sceneWorkBatches)
{
    std::vector<GPUMetaSurfaceGeneratorI*> hMetaSurfaceGenGPUPtrs;
    TracerError err = TracerError::OK;
    // Generate Material Interfaces
    for(const auto& matGroup : scene.MaterialGroups())
    {
        matGroup.second->GeneratePerMaterialInterfaces();
    }
    // Generate Primitive SoA Data on the GPU
    for(const auto& primGroup : scene.PrimitiveGroups())
    {
        primGroup.second->GeneratePrimDataGPUPtr();
    }

    // Work Batch Ids are not linear (some tracers may not use some work
    // batches). For example, a camera->light path tracer
    // does not need to create camera batches etc.
    // Loop the work batches first to find the max amount of batches
    uint32_t totalBatchCount = 0;
    for(const auto& workList : sceneWorkBatches)
    {
        uint32_t batchId = workList.first;
        totalBatchCount = std::max(batchId, totalBatchCount);
    }
    hMetaSurfaceGenGPUPtrs.resize(totalBatchCount + 1, nullptr);

    // Now ask work batches to generate the MetaSurfaceGenerator class
    for(const auto& workList : sceneWorkBatches)
    {
        uint32_t batchId = workList.first;
        const std::vector<GPUWorkBatchI*>& works = workList.second;
        for(GPUWorkBatchI* work : works)
        {
            GPUMetaSurfaceGeneratorI* metaSurfaceGenerator;
            if((err = work->CreateMetaSurfaceGenerator(metaSurfaceGenerator)) != TracerError::OK)
                return err;
            hMetaSurfaceGenGPUPtrs.at(batchId) = metaSurfaceGenerator;
        }
    }

    GPUMemFuncs::AllocateMultiData(std::tie(dGeneratorInterfaces),
                                   generatorMem, {hMetaSurfaceGenGPUPtrs.size()});
    CUDA_CHECK(cudaMemcpy(dGeneratorInterfaces, hMetaSurfaceGenGPUPtrs.data(),
                          sizeof(GPUMetaSurfaceGeneratorI*) * hMetaSurfaceGenGPUPtrs.size(),
                          cudaMemcpyHostToDevice));
    return TracerError::OK;
}