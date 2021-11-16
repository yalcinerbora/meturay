#include "GPUAcceleratorOptix.cuh"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"
#include "CudaSystem.hpp"

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
    // This call is not used for OptiX
    // throw an execption if it is called
    throw TracerException(TracerError::TRACER_INTERNAL_ERROR);
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
    //for(const auto& [surfId, aabb] : aabbMap)
    //{
    //    METU_LOG(surfId);
    //}

    //// Generate Temporary Instance
    //DeviceMemory(sizeof(OptixInstance) * aabbMap.size());

    //// Call Kernel

    //OptixInstanceFlags flags = OPTIX_INSTANCE_FLAG_NONE;
    //OptixInstance instance = {};
    //instance.flags = flags;
    //instance.instanceId = 0;
    //instance.sbtOffset = 0;
    //instance.visibilityMask = 0xFF;
    //instance.transform = {1, 0, 0, 0,
    //                      0, 1, 0, 0,
    //                      0, 0, 1, 0};
    //instance.traversableHandle = 0;




    //{
    //    float transform[12];
    //};


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

    uint64_t totalVertexCount = this->primitiveGroup.TotalDataCount();
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

    //===============================//
    //  ACTUAL TRAVERSAL GENERATION  //
    //===============================//
    uint32_t deviceIndex = 0;
    for(const auto& [gpu, optixContext] : optixSystem->OptixCapableDevices())
    {
        CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
        DeviceTraversables& gpuTraverseData = optixDataPerGPU[deviceIndex];

        uint32_t buildInputCount = 0;
        std::array<OptixBuildInput, SceneConstants::MaxPrimitivePerSurface> buildInputs;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const auto& range = primRangeList[i];
            if(range[0] == std::numeric_limits<uint64_t>::max())
                break;

            buildInputCount++;

            uint32_t geometryFlag = OPTIX_GEOMETRY_FLAG_NONE;

            uint32_t indexCount = static_cast<uint32_t>(primRangeList[i][1] - primRangeList[i][0]);
            uint32_t offset = static_cast<uint32_t>(primRangeList[i][0]);

            OptixBuildInput& buildInput = buildInputs[i];
            buildInput = OptixBuildInput{};
            buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            // Vertex
            CUdeviceptr dPositions = AsOptixPtr(primData.positions);
            CUdeviceptr dIndices = AsOptixPtr(primData.indexList);
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
            optixContext,
            &accelOptions, buildInputs.data(),
            buildInputCount, &accelMemorySizes
        ));

        // Allocate Temp Buffer for Build
        Byte* dTempBuild;
        uint64_t* dCompactedSize;
        Byte* dTempMem;
        DeviceLocalMemory tempBuildBuffer(&gpu);
        GPUMemFuncs::AllocateMultiData(std::tie(dTempBuild, dCompactedSize), tempBuildBuffer,
                                        {accelMemorySizes.outputSizeInBytes, 1}, 128);
        DeviceLocalMemory tempMem(&gpu, accelMemorySizes.tempSizeInBytes);
        dTempMem = static_cast<Byte*>(tempMem);


        // While building fetch compacted output size
        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = AsOptixPtr(dCompactedSize);

        OptixTraversableHandle traversable;
        OPTIX_CHECK(optixAccelBuild(optixContext, (cudaStream_t)0,
                                    &accelOptions,
                                    // Build Inputs
                                    buildInputs.data(), buildInputCount,
                                    // Temp Memory
                                    AsOptixPtr(dTempMem), accelMemorySizes.tempSizeInBytes,
                                    // Output Memory
                                    AsOptixPtr(dTempBuild), accelMemorySizes.outputSizeInBytes,
                                    &traversable, &emitProperty, 1));

        // Get compacted size to CPU
        uint64_t hCompactAccelSize;
        CUDA_CHECK(cudaMemcpy(&hCompactAccelSize, dCompactedSize,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost));

        if(hCompactAccelSize < tempBuildBuffer.Size())
        {
            DeviceLocalMemory compactedMemory(&gpu, hCompactAccelSize);

            // use handle as input and output
            OPTIX_CHECK(optixAccelCompact(optixContext, (cudaStream_t)0,
                                          traversable,
                                          AsOptixPtr(compactedMemory),
                                          hCompactAccelSize,
                                          &traversable));

            gpuTraverseData.tMemories[innerIndex] = std::move(compactedMemory);
        }
        else
            gpuTraverseData.tMemories[innerIndex] = std::move(tempBuildBuffer);

        gpuTraverseData.traversables[innerIndex] = traversable;
        deviceIndex++;
    }

    // All Done!
    return TracerError::OK;
}

void GPUBaseAcceleratorOptiX::SetOptiXSystem(const OptiXSystem* sys)
{
    optixSystem = sys;

}

OptixTraversableHandle GPUBaseAcceleratorOptiX::GetBaseTraversable() const
{
    return baseTraversable;
}

// Accelerator Instancing for basic primitives
template class GPUAccOptiXGroup<GPUPrimitiveTriangle>;
template class GPUAccOptiXGroup<GPUPrimitiveSphere>;

static_assert(IsTracerClass<GPUAccOptiXGroup<GPUPrimitiveTriangle>>::value,
              "GPUAccOptixGroup<GPUPrimitiveTriangle> is not a Tracer Class.");
static_assert(IsTracerClass<GPUAccOptiXGroup<GPUPrimitiveSphere>>::value,
              "GPUAccOptixGroup<GPUPrimitiveSphere> is not a Tracer Class.");