#pragma once

#include "GPULightP.cuh"
#include "GPUTransformI.h"
#include "TypeTraits.h"
#include "MangledNames.h"

class GPULightNull final : public GPULightP
{
    private:
    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightNull(// Base Class Related
                                             const TextureRefI<2, Vector3f>& gRad,
                                             uint16_t mediumId, HitKey,
                                             const GPUTransformI& gTrans);
                                ~GPULightNull() = default;
        // Interface
        __device__ void         Sample(// Output
                                       float& distance,
                                       Vector3& direction,
                                       float& pdf,
                                       // Input
                                       const Vector3& worldLoc,
                                       // I-O
                                       RandomGPU&) const override;

        __device__ void         GenerateRay(// Output
                                            RayReg&,
                                            // Input
                                            const Vector2i& sampleId,
                                            const Vector2i& sampleMax,
                                            // I-O
                                            RandomGPU&,
                                            // Options
                                            bool antiAliasOn = true) const override;

        __device__ float        Pdf(const Vector3& direction,
                                    const Vector3& position) const override;

        __device__ bool         CanBeSampled() const override;
};

class CPULightGroupNull final : public CPULightGroupP<GPULightNull>
{
    public:
        TYPENAME_DEF(LightGroup, "Null");

        static constexpr const char*    POSITION_NAME = "position";

        using Base = CPULightGroupP<GPULightNull>;

    private:
    protected:
    public:
        // Cosntructors & Destructor
                                    CPULightGroupNull(const GPUPrimitiveGroupI*,
                                                       const CudaGPU&);
                                    ~CPULightGroupNull() = default;

        const char*				    Type() const override;
		SceneError				    InitializeGroup(const EndpointGroupDataList& endpointNodes,
                                                    const TextureNodeMap& textures,
                                                    const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                    const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                    uint32_t batchId, double time,
                                                    const std::string& scenePath) override;
        SceneError				    ChangeTime(const NodeListing& lightNodes, double time,
                                               const std::string& scenePath) override;
        TracerError				    ConstructEndpoints(const GPUTransformI**,
                                                       const CudaSystem&) override;
};

__device__
inline GPULightNull::GPULightNull(// Base Class Related
                                  const TextureRefI<2, Vector3f>& gRad,
                                  uint16_t mediumId, HitKey hk,
                                  const GPUTransformI& gTrans)
    : GPULightP(gRad, mediumIndex, hk, gTransform)
{}

__device__
inline void GPULightNull::Sample(// Output
                                 float& distance,
                                 Vector3& direction,
                                 float& pdf,
                                 // Input
                                 const Vector3& worldLoc,
                                 // I-O
                                 RandomGPU&) const
{
    distance = FLT_MAX;
    direction = Zero3f;
    pdf = 0.0f;
}

__device__
inline void GPULightNull::GenerateRay(// Output
                                      RayReg&,
                                      // Input
                                      const Vector2i& sampleId,
                                      const Vector2i& sampleMax,
                                      // I-O
                                      RandomGPU&,
                                      // Options
                                      bool antiAliasOn) const
{}

__device__
inline float GPULightNull::Pdf(const Vector3& worldDir,
                               const Vector3& worldPos) const
{
    return 0.0f;
}

__device__
inline bool GPULightNull::CanBeSampled() const
{
    return false;
}

inline CPULightGroupNull::CPULightGroupNull(const GPUPrimitiveGroupI* pg,
                                            const CudaGPU& gpu)
    : Base(*pg, gpu)
{}

inline const char* CPULightGroupNull::Type() const
{
    return TypeName();
}

__global__
static void KCConstructGPULightNull(GPULightNull* gLightLocations,
                                         //
                                    const TextureRefI<2, Vector3f>** gRads,
                                    const uint16_t* gMediumIndices,
                                    const HitKey* gWorkKeys,
                                    const TransformId* gTransformIds,
                                    //
                                    const GPUTransformI** gTransforms,
                                    uint32_t lightCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < lightCount;
        globalId += blockDim.x * gridDim.x)
    {
        new (gLightLocations + globalId) GPULightNull(*gRads[globalId],
                                                       gMediumIndices[globalId],
                                                       gWorkKeys[globalId],
                                                       *gTransforms[gTransformIds[globalId]]);
    }
}

inline SceneError CPULightGroupNull::InitializeGroup(const EndpointGroupDataList& lightNodes,
                                                     const TextureNodeMap& textures,
                                                     const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                     const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                     uint32_t batchId, double time,
                                                     const std::string& scenePath)
{
    SceneError e = SceneError::OK;

    if((e = InitializeCommon(lightNodes, textures,
                             mediumIdIndexPairs,
                             transformIdIndexPairs,
                             batchId, time,
                             scenePath)) != SceneError::OK)
        return e;
    return SceneError::OK;
}

inline SceneError CPULightGroupNull::ChangeTime(const NodeListing& lightNodes, double time,
                                                const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

inline TracerError CPULightGroupNull::ConstructEndpoints(const GPUTransformI** dGlobalTransformArray,
                                                         const CudaSystem&)
{
    TracerError e = TracerError::OK;
    // Construct Texture References
    if((e = ConstructTextureReferences()) != TracerError::OK)
        return e;

    // Gen Temporary Memory
    DeviceMemory tempMemory;

    const uint16_t* dMediumIndices;
    const TransformId* dTransformIds;
    const HitKey* dWorkKeys;
    DeviceMemory::AllocateMultiData(std::tie(dMediumIndices, dTransformIds, dWorkKeys),
                                    tempMemory,
                                    {lightCount, lightCount, lightCount});

    // Set a GPU
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    // Load Data to Temp Memory
    CUDA_CHECK(cudaMemcpy(const_cast<uint16_t*>(dMediumIndices),
                          hMediumIds.data(),
                          sizeof(uint16_t) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<TransformId*>(dTransformIds),
                          hTransformIds.data(),
                          sizeof(TransformId) * lightCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(const_cast<HitKey*>(dWorkKeys),
                          hWorkKeys.data(),
                          sizeof(HitKey) * lightCount,
                          cudaMemcpyHostToDevice));
    // Call allocation kernel
    gpu.GridStrideKC_X(0, 0,
                       lightCount,
                       //
                       KCConstructGPULightNull,
                       //
                       const_cast<GPULightNull*>(dGPULights),
                       //
                       dRadiances,
                       dMediumIndices,
                       dWorkKeys,
                       dTransformIds,
                       //
                       dGlobalTransformArray,
                       lightCount);

    gpu.WaitMainStream();

    SetLightLists();

    return TracerError::OK;
}

static_assert(IsTracerClass<CPULightGroupNull>::value,
              "CPULightGroupNull is not a tracer class");