#include "GPUAcceleratorOptix.cuh"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"
#include "CudaSystem.hpp"

#include <numeric>

template <class PGroup>
__global__
static void KCCopyPrimPositions(Vector3f* gPositions,
                                // Iputs
                                typename PGroup::PrimitiveData pData,
                                const Vector2ul primRanges)
{
    const uint32_t primCount = primRanges[1] - primRanges[0];

    // Local Position Registers
    constexpr uint32_t POS_PER_PRIM = PGroup::PositionPerPrim;
    Vector3f positions[POS_PER_PRIM];

    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        // Read / Write Locations
        uint32_t readIndex = globalId + primRanges[0];
        uint32_t writeIndex = globalId;
        // Copy to local registers
        PGroup::AcquirePositions(positions, readIndex, pData);
        // Write to global mem
        Vector3f* gPositionsLocal = gPositions + POS_PER_PRIM * writeIndex;
        #pragma unroll
        for(uint32_t i = 0; i < POS_PER_PRIM; i++)
        {
            gPositionsLocal[i] = positions[i];
        }
        printf("Writing (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)\n",
               gPositionsLocal[0][0], gPositionsLocal[0][1], gPositionsLocal[0][2],
               gPositionsLocal[1][0], gPositionsLocal[1][1], gPositionsLocal[1][2],
               gPositionsLocal[2][0], gPositionsLocal[2][1], gPositionsLocal[2][2]);
    }
}

__global__
static void KCPopulateTransforms(// I-O
                                 OptixInstance* gInstances,
                                 // Inputs
                                 const GPUTransformI** gGlobalTransformArray,
                                 const TransformId* gTransformIds,
                                 const PrimTransformType* gTransformTypes,
                                 //
                                 uint32_t instanceCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < instanceCount; globalId += blockDim.x * gridDim.x)
    {
        Matrix4x4 transform = Indentity4x4;
        if(gTransformTypes[globalId] == PrimTransformType::CONSTANT_LOCAL_TRANSFORM)
        {
            const GPUTransformI* gTransform = gGlobalTransformArray[gTransformIds[globalId]];
            transform = gTransform->GetLocalToWorldAsMatrix();
            // Matrix4x4 is column-major matrix OptiX wants row-major
            transform.TransposeSelf();
        }
        else
        {
            // This means that transform cannot be applied to the primitives constantly and
            // accelerator is generated using that transform so just initialize with identity
            transform = Indentity4x4;
        }
        // Can nvcc optimize this?
        memcpy(gInstances[globalId].transform,
               &transform, sizeof(float) * 12);
    }
}

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

void GPUBaseAcceleratorOptiX::GetReady(const CudaSystem&, uint32_t)
{

}

void GPUBaseAcceleratorOptiX::Hit(const CudaSystem&,
                                  // Output
                                  HitKey*,
                                  // Inputs
                                  const RayGMem*,
                                  const RayId*,
                                  const uint32_t) const
{
    // This call is not used for OptiX
    // throw an execption if it is called
    throw TracerException(TracerError::TRACER_INTERNAL_ERROR);
}

SceneError GPUBaseAcceleratorOptiX::Initialize(// Accelerator Option Node
                                               const SceneNodePtr&,
                                               // List of surface to leaf accelerator ids
                                               const std::map<uint32_t, HitKey>& keyMap)
{
    idLookup.clear();
    //leafs.resize(keyMap.size());

    uint32_t i = 0;
    for(const auto& pair : keyMap)
    {
        //leafs[i].accKey = pair.second;
        idLookup.emplace(pair.first, i);
        i++;
    }
    return SceneError::OK;
}

TracerError GPUBaseAcceleratorOptiX::Construct(const CudaSystem&,
                                               // List of surface AABBs
                                               const SurfaceAABBList&)
{
    // This function is not used when OptiX is enabled
    return TracerError::TRACER_INTERNAL_ERROR;
}


TracerError GPUBaseAcceleratorOptiX::Construct(const std::vector<std::vector<OptixTraversableHandle>>& gpuTraversables,
                                               const std::vector<PrimTransformType>& hTransformTypes,
                                               const TransformId* dAllTransformIds,
                                               const GPUTransformI** dGlobalTransformArray)
{
    assert(hTransformTypes.size() == idLookup.size());
    assert(idLookup.size() == gpuTraversables.front().size());

    DeviceMemory transformTempMemory;
    PrimTransformType* dTransformTypes;
    GPUMemFuncs::AllocateMultiData(std::tie(dTransformTypes), transformTempMemory,
                                   {hTransformTypes.size()});
    CUDA_CHECK(cudaMemcpy(dTransformTypes, hTransformTypes.data(),
                          hTransformTypes.size() * sizeof(PrimTransformType),
                          cudaMemcpyHostToDevice));

    //===============================//
    //  ACTUAL TRAVERSAL GENERATION  //
    //===============================//
    uint32_t deviceIndex = 0;
    for(const auto& [gpu, optixContext] : optixSystem->OptixCapableDevices())
    {
        CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
        auto& optixData = optixGPUData[deviceIndex];
        const auto& traversables = gpuTraversables[deviceIndex];

        // Allocate Temp
        std::vector<OptixInstance> hInstances;
        hInstances.reserve(idLookup.size());
        for(uint32_t i = 0; i < idLookup.size(); i++)
        {
            OptixInstance instance = {};
            instance.traversableHandle = traversables[i];
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.instanceId = i;
            instance.sbtOffset = i;
            instance.visibilityMask = 0xFF;
            // Leave Transform for now

            hInstances.push_back(instance);
        }
        DeviceLocalMemory tempInstanceMemory(&gpu, hInstances.size() * sizeof(OptixInstance));
        OptixInstance* dInstances = static_cast<OptixInstance*>(tempInstanceMemory);

        // Copy Instances to GPU
        CUDA_CHECK(cudaMemcpy(dInstances, hInstances.data(),
                              hInstances.size() * sizeof(OptixInstance),
                              cudaMemcpyHostToDevice));

        // Copy transforms (as matrices to the instance data)
        gpu.GridStrideKC_X(0, (cudaStream_t)0, hInstances.size(),
                           //
                           KCPopulateTransforms,
                            // I-O
                           dInstances,
                           // Inputs
                           dGlobalTransformArray,
                           dAllTransformIds,
                           dTransformTypes,
                           //
                           static_cast<uint32_t>(hInstances.size()));

        OptixBuildInput buildInput = OptixBuildInput{};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInput.instanceArray.instances = AsOptixPtr(dInstances);
        buildInput.instanceArray.numInstances = static_cast<uint32_t>(hInstances.size());
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes accelMemorySizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage
        (
            optixContext,
            &accelOptions, &buildInput, 1,
            &accelMemorySizes
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
                                    &buildInput, 1,
                                    // Temp Memory
                                    AsOptixPtr(dTempMem), accelMemorySizes.tempSizeInBytes,
                                    // Output Memory
                                    AsOptixPtr(dTempBuild), accelMemorySizes.outputSizeInBytes,
                                    &traversable, &emitProperty, 1));
        CUDA_KERNEL_CHECK();

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
            CUDA_KERNEL_CHECK();

            optixData.tMemory = std::move(compactedMemory);
        }
        else
            optixData.tMemory = std::move(tempBuildBuffer);
        optixData.traversable = traversable;
        deviceIndex++;
    }

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
    TracerError err = TracerError::OK;
    if((err = FillLeaves(system, surface)) != TracerError::OK)
        return err;

    using LeafData = typename GPUPrimitiveTriangle::LeafData;
    using PrimitiveData = typename GPUPrimitiveTriangle::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(this->primitiveGroup);

    uint32_t innerIndex = idLookup.at(surface);
    const PrimitiveRangeList& primRangeList = primitiveRanges[innerIndex];
    const PrimitiveIdList& primIdList = primitiveIds[innerIndex];
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

        // IMPORTANT
        // Optix wants the triangles & Indices to be in a memory
        // that is allocated with a "cudaMalloc" function
        // Current METUray manages its memory using "cudaMallocManaged"
        // for common memory except textures (which uses cudaArray allocation and
        // it is not multi-device capable anyway)
        // We need to copy data to device local memory for construction
        // after that traversable does not refer to these memory so we can
        // delete
        // Acquire vertex sizes
        size_t surfacePrimCount = 0;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const auto& range = primRangeList[i];
            if(range[0] == std::numeric_limits<uint64_t>::max())
                break;
            // Use primitive count as vertex count
            surfacePrimCount += range[1] - range[0];
        }

        // Allocate Local Memory
        DeviceLocalMemory localVertexMem(&gpu, (sizeof(Vector3f) * surfacePrimCount *
                                                GPUPrimitiveTriangle::PositionPerPrim));
        Vector3f* dPositions = static_cast<Vector3f*>(localVertexMem);
        size_t offset = 0;
        // Copy to Local Memory
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const auto& range = primRangeList[i];
            if(range[0] == std::numeric_limits<uint64_t>::max())
                break;

            size_t subRangePrimCount = range[1] - range[0];
            gpu.GridStrideKC_X(0, (cudaStream_t)0, subRangePrimCount,
                               //
                               KCCopyPrimPositions<GPUPrimitiveTriangle>,
                               //
                               dPositions + offset,
                               primData,
                               range);
            offset += subRangePrimCount;
        }
        assert(offset == surfacePrimCount);

        // Generate build input now
        offset = 0;
        uint32_t buildInputCount = 0;
        std::array<OptixBuildInput, SceneConstants::MaxPrimitivePerSurface> buildInputs;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const auto& range = primRangeList[i];
            if(range[0] == std::numeric_limits<uint64_t>::max())
                break;
            buildInputCount++;

            // Gen Vertex Ptr
            CUdeviceptr vertices = AsOptixPtr(dPositions + offset);
            size_t localPrimCount = range[1] - range[0];
            offset += localPrimCount;
            size_t vertexCount = GPUPrimitiveTriangle::PositionPerPrim * localPrimCount;

            // Enable/Disable Any hit if batch has alpha map
            uint32_t geometryFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
            if(primitiveGroup.PrimitiveHasAlphaMap(primIdList[i]))
                geometryFlags = OPTIX_GEOMETRY_FLAG_NONE;

            OptixBuildInput& buildInput = buildInputs[i];
            buildInput = {};
            buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            // Vertex
            buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            buildInput.triangleArray.numVertices = static_cast<uint32_t>(vertexCount);
            buildInput.triangleArray.vertexBuffers = &vertices;
            buildInput.triangleArray.vertexStrideInBytes = sizeof(Vector3f);
            // Index (we dont use indices)
            //buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
            //buildInput.triangleArray.numIndexTriplets = 0;
            //buildInput.triangleArray.indexBuffer = 0;
            //buildInput.triangleArray.indexStrideInBytes = 0;
            //buildInput.triangleArray.primitiveIndexOffset = 0;
            // SBT
            buildInput.triangleArray.flags = &geometryFlags;
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
        DeviceLocalMemory buildBuffer(&gpu, accelMemorySizes.outputSizeInBytes);
        Byte* dTempBuild = static_cast<Byte*>(buildBuffer);
        Byte* dTemp;
        uint64_t* dCompactedSize;
        DeviceLocalMemory tempMemory(&gpu);
        GPUMemFuncs::AllocateMultiData(std::tie(dTemp, dCompactedSize), tempMemory,
                                       {accelMemorySizes.tempSizeInBytes, 1}, 128);


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
                                    AsOptixPtr(dTemp), accelMemorySizes.tempSizeInBytes,
                                    // Output Memory
                                    AsOptixPtr(dTempBuild), accelMemorySizes.outputSizeInBytes,
                                    &traversable, &emitProperty, 1));
        CUDA_KERNEL_CHECK();

        // Get compacted size to CPU
        uint64_t hCompactAccelSize;
        CUDA_CHECK(cudaMemcpy(&hCompactAccelSize, dCompactedSize,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost));

        if(hCompactAccelSize < buildBuffer.Size())
        {
            DeviceLocalMemory compactedMemory(&gpu, hCompactAccelSize);

            // use handle as input and output
            OPTIX_CHECK(optixAccelCompact(optixContext, (cudaStream_t)0,
                                          traversable,
                                          AsOptixPtr(compactedMemory),
                                          hCompactAccelSize,
                                          &traversable));
            CUDA_KERNEL_CHECK();

            gpuTraverseData.tMemories[innerIndex] = std::move(compactedMemory);
        }
        else
            gpuTraverseData.tMemories[innerIndex] = std::move(buildBuffer);

        gpuTraverseData.traversables[innerIndex] = traversable;
        deviceIndex++;
    }

    // All Done!
    return TracerError::OK;
}

void GPUBaseAcceleratorOptiX::SetOptiXSystem(const OptiXSystem* sys)
{
    optixSystem = sys;
    optixGPUData.reserve(sys->OptixCapableDevices().size());
    for(const auto& [gpu, optixCosntext] : sys->OptixCapableDevices())
    {
        optixGPUData.push_back({DeviceLocalMemory(&gpu), 0});
    }
}

OptixTraversableHandle GPUBaseAcceleratorOptiX::GetBaseTraversable(int optixGPUIndex) const
{
    return optixGPUData[optixGPUIndex].traversable;
}

// Accelerator Instancing for basic primitives
template class GPUAccOptiXGroup<GPUPrimitiveTriangle>;
template class GPUAccOptiXGroup<GPUPrimitiveSphere>;

static_assert(IsTracerClass<GPUAccOptiXGroup<GPUPrimitiveTriangle>>::value,
              "GPUAccOptixGroup<GPUPrimitiveTriangle> is not a Tracer Class.");
static_assert(IsTracerClass<GPUAccOptiXGroup<GPUPrimitiveSphere>>::value,
              "GPUAccOptixGroup<GPUPrimitiveSphere> is not a Tracer Class.");