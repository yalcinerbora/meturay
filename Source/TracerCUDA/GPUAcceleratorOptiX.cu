#include "GPUAcceleratorOptix.cuh"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"

const char* GPUBaseAcceleratorOptiX::TypeName()
{
    return "OptiX";
}

GPUBaseAcceleratorOptiX::GPUBaseAcceleratorOptiX()
    : optixSystem(nullptr)
{}

const char* GPUBaseAcceleratorOptiX::Type() const
{
    return TypeName();
}

TracerError GPUBaseAcceleratorOptiX::LoadModule()
{
    OptixModuleCompileOptions moduleCompileOpts = {};
    OptixPipelineCompileOptions pipelineCompileOpts = {};
    moduleCompileOpts.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    if constexpr(METU_DEBUG_BOOL)
    {
        moduleCompileOpts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        pipelineCompileOpts.exceptionFlags = (OPTIX_EXCEPTION_FLAG_DEBUG |
                                              OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                              OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW);
    }
    else
    {
        moduleCompileOpts.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
        pipelineCompileOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    }
    pipelineCompileOpts.usesMotionBlur = false;
    pipelineCompileOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOpts.numPayloadValues      = 2;
    pipelineCompileOpts.numAttributeValues    = 2;
    pipelineCompileOpts.pipelineLaunchParamsVariableName = "params";

    TracerError err = TracerError::OK;
    if((err = optixSystem->OptixGenerateModules(moduleCompileOpts, pipelineCompileOpts,
                                                MODULE_BASE_NAME)) != TracerError::OK)
        return err;
    return TracerError::OK;
}

void GPUBaseAcceleratorOptiX::GetReady(const CudaSystem& system,
                                       uint32_t rayCount)
{

}

void GPUBaseAcceleratorOptiX::Hit(const CudaSystem& system,
                                // Output
                                  HitKey* dAcceleratorKeys,
                                  // Inputs
                                  const RayGMem* dRays,
                                  const RayId* dRayIds,
                                  const uint32_t rayCount) const
{

}

SceneError GPUBaseAcceleratorOptiX::Initialize(// Accelerator Option Node
                                               const SceneNodePtr& node,
                                               // List of surface to leaf accelerator ids
                                               const std::map<uint32_t, HitKey>& keyMap)
{
    idLookup.clear();
    leafs.resize(keyMap.size());

    uint32_t i = 0;
    for(const auto& pair : keyMap)
    {
        leafs[i].accKey = pair.second;
        idLookup.emplace(pair.first, i);
        i++;
    }
    return SceneError::OK;
}

TracerError GPUBaseAcceleratorOptiX::Constrcut(const CudaSystem&,
                                             // List of surface AABBs
                                             const SurfaceAABBList& aabbMap)
{

    return TracerError::OK;
}

TracerError GPUBaseAcceleratorOptiX::Destruct(const CudaSystem&)
{
    // TODO: Implement?
    return TracerError::OK;
}

const AABB3f& GPUBaseAcceleratorOptiX::SceneExtents() const
{
    return sceneAABB;
}

template<>
TracerError GPUAccOptiXGroup<GPUPrimitiveTriangle>::ConstructAccelerator(uint32_t surface,
                                                                         const CudaSystem& system)
{
    // Specialized Triangle Primitive Function

    uint64_t totalVertexCount = this->primitiveGroup.TotalDataCount() * 3;
    // Currently optix only supports 32-bit indices
    if(totalVertexCount >= std::numeric_limits<uint32_t>::max())
        return TracerError::UNABLE_TO_CONSTRUCT_ACCELERATOR;

    using LeafData = typename GPUPrimitiveTriangle::LeafData;
    using PrimitiveData = typename GPUPrimitiveTriangle::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(this->primitiveGroup);

    uint32_t innerIndex = idLookup.at(surface);
    const PrimitiveRangeList& primRangeList = primitiveRanges[innerIndex];
    const PrimTransformType tType = this->primitiveGroup.TransformType();

    // Select Transform for construction
    const GPUTransformI* worldTransform = nullptr;
    AcquireAcceleratorGPUTransform(worldTransform,
                                   dAccTransformIds,
                                   this->dTransforms,
                                   innerIndex,
                                   system.BestGPU());
    const GPUTransformI* transform = worldTransform;
    if(tType == PrimTransformType::CONSTANT_LOCAL_TRANSFORM)
    {
        AcquireIdentityTransform(transform,
                                 this->dTransforms,
                                 this->identityTransformIndex,
                                 system.BestGPU());
    }

    uint32_t buildInputCount = 0;
    std::array<OptixBuildInput, SceneConstants::MaxPrimitivePerSurface> buildInputs;
    for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
    {
        buildInputCount++;

        const auto& range = primRangeList[i];
        if(range[0] == std::numeric_limits<uint64_t>::max())
            break;

        uint32_t geometryFlag = OPTIX_GEOMETRY_FLAG_NONE;

        uint32_t indexCount = static_cast<uint32_t>(primRangeList[i][1] - primRangeList[i][0]);
        uint32_t offset = static_cast<uint32_t>(primRangeList[i][0]);

        OptixBuildInput& buildInput = buildInputs[i];
        buildInput = OptixBuildInput{};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        // Vertex
        CUdeviceptr dPositions = reinterpret_cast<CUdeviceptr>(primData.positions);
        CUdeviceptr dIndices = reinterpret_cast<CUdeviceptr>(primData.indexList);
        // Vertex
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.numVertices = static_cast<uint32_t>(totalVertexCount);
        buildInput.triangleArray.vertexBuffers = &dPositions;
        buildInput.triangleArray.vertexStrideInBytes = sizeof(Vector3f);
        // Index
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.numIndexTriplets = indexCount;
        buildInput.triangleArray.indexBuffer = dIndices;
        // We store 64 bit values on index make stride as such
        buildInput.triangleArray.indexStrideInBytes = sizeof(uint64_t) * 3;
        buildInput.triangleArray.primitiveIndexOffset = offset;
        // SBT
        buildInput.triangleArray.flags = &geometryFlag;
        buildInput.triangleArray.numSbtRecords = 1;
        buildInput.triangleArray.sbtIndexOffsetBuffer = 0;
        buildInput.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        buildInput.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
    }

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes accelMemorySizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
    (
        optiXSystem->OptixContext(system.BestGPU()),
        &accelOptions, buildInputs.data(),
        buildInputCount, &accelMemorySizes
    ));





    uint64_t* dCompactedSize;
    Byte* dTempBuild;

    DeviceMemory tempBuildBuffer;
    DeviceMemory::AllocateMultiData(std::tie(dCompactedSize, dTempBuild), tempBuildBuffer,
                                    {1, accelMemorySizes.outputSizeInBytes});
    DeviceMemory tempMem(accelMemorySizes.tempSizeInBytes);

    // While building fetch compacted output size
    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = reinterpret_cast<CUdeviceptr>(dCompactedSize);

    OptixTraversableHandle traversable;
    OPTIX_CHECK(optixAccelBuild(optiXSystem->OptixContext(system.BestGPU()),
                                0,                                  // CUDA stream
                                &accelOptions,
                                buildInputs.data(),
                                buildInputCount,
                                reinterpret_cast<CUdeviceptr>(static_cast<Byte*>(tempMem)), tempMem.Size(),
                                reinterpret_cast<CUdeviceptr>(dTempBuild), accelMemorySizes.outputSizeInBytes,
                                &traversable, &emitProperty, 1));


    //CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    //CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));

    // Get compacted size
    size_t hCompactAccelSize;
    CUDA_CHECK(cudaMemcpy(&hCompactAccelSize, dCompactedSize,
                          sizeof(size_t), cudaMemcpyDeviceToHost));

    if(hCompactAccelSize < accelMemorySizes.outputSizeInBytes)
    {
        //CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(optiXSystem->OptixContext(system.BestGPU()), 0,
                                      traversable, /*CHANGE THIS*/0,
                                      hCompactAccelSize,
                                      &traversable));

        //CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    //else
    //{
    //    state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    //}





    return TracerError::UNABLE_TO_CONSTRUCT_ACCELERATOR;
}

const OptiXSystem* GPUBaseAcceleratorOptiX::GetOptiXSystem(const OptiXSystem* sys) const
{
    return optixSystem.get();
}

// Accelerator Instancing for basic primitives
template class GPUAccOptiXGroup<GPUPrimitiveTriangle>;
template class GPUAccOptiXGroup<GPUPrimitiveSphere>;

static_assert(IsTracerClass<GPUAccOptiXGroup<GPUPrimitiveTriangle>>::value,
              "GPUAccOptixGroup<GPUPrimitiveTriangle> is not a Tracer Class.");
static_assert(IsTracerClass<GPUAccOptiXGroup<GPUPrimitiveSphere>>::value,
              "GPUAccOptixGroup<GPUPrimitiveSphere> is not a Tracer Class.");