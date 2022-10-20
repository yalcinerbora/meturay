#pragma once

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

            hMetaSurfaceGenGPUPtrs.push_back(metaSurfaceGenerator);
        }
    }

    GPUMemFuncs::AllocateMultiData(std::tie(dGeneratorInterfaces),
                                   generatorMem, {hMetaSurfaceGenGPUPtrs.size()});
    CUDA_CHECK(cudaMemcpy(dGeneratorInterfaces, hMetaSurfaceGenGPUPtrs.data(),
                          sizeof(GPUMetaSurfaceGeneratorI*) * hMetaSurfaceGenGPUPtrs.size(),
                          cudaMemcpyHostToDevice));
}