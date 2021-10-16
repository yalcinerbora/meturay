#pragma once

#include "GPULightP.cuh"
#include "GPUTransformI.h"
#include "TypeTraits.h"
#include "MangledNames.h"

class GPULightConstant final : public GPULightP
{
    private:
    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightConstant(// Base Class Related
                                                 const TextureRefI<2, Vector3f>& gRad,
                                                 uint16_t mediumId, HitKey,
                                                 const GPUTransformI& gTrans);
                                ~GPULightConstant() = default;
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
        __device__ float        Pdf(float distance,
                                    const Vector3& hitPosition,
                                    const Vector3& direction,
                                    const QuatF& tbnRotation) const override;

        __device__ bool         CanBeSampled() const override;
};

class CPULightGroupConstant final : public CPULightGroupP<GPULightConstant>
{
    public:
        TYPENAME_DEF(LightGroup, "Constant");

        using Base = CPULightGroupP<GPULightConstant>;

    private:
    protected:
    public:
        // Cosntructors & Destructor
                                    CPULightGroupConstant(const GPUPrimitiveGroupI*,
                                                       const CudaGPU&);
                                    ~CPULightGroupConstant() = default;

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
inline GPULightConstant::GPULightConstant(// Base Class Related
                                          const TextureRefI<2, Vector3f>& gRad,
                                          uint16_t mediumId, HitKey hk,
                                          const GPUTransformI& gTrans)
    : GPULightP(gRad, mediumIndex, hk, gTransform)
{}

__device__
inline void GPULightConstant::Sample(// Output
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
inline void GPULightConstant::GenerateRay(// Output
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
inline float GPULightConstant::Pdf(const Vector3& worldDir,
                                   const Vector3& worldPos) const
{
    return 0.0f;
}

__device__
inline float GPULightConstant::Pdf(float distance,
                                   const Vector3& hitPosition,
                                   const Vector3& direction,
                                   const QuatF& tbnRotation) const
{
    return 0.0f;
}

__device__
inline bool GPULightConstant::CanBeSampled() const
{
    return false;
}

inline CPULightGroupConstant::CPULightGroupConstant(const GPUPrimitiveGroupI* pg,
                                                    const CudaGPU& gpu)
    : Base(*pg, gpu)
{}

inline const char* CPULightGroupConstant::Type() const
{
    return TypeName();
}

__global__
static void KCConstructGPULightConstant(GPULightConstant* gLightLocations,
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
        new (gLightLocations + globalId) GPULightConstant(*gRads[globalId],
                                                          gMediumIndices[globalId],
                                                          gWorkKeys[globalId],
                                                          *gTransforms[gTransformIds[globalId]]);
    }
}

inline SceneError CPULightGroupConstant::InitializeGroup(const EndpointGroupDataList& lightNodes,
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

    // This object does not expose GPU Groups to the system
    // since it is not clearly defined (it only returns radiance value)
    // its pdf / sample routines are not valid
    // so NEE estimator etc. does not tries to sample this type of light
    gpuLightList.clear();
    gpuEndpointList.clear();

    return SceneError::OK;
}

inline SceneError CPULightGroupConstant::ChangeTime(const NodeListing& lightNodes, double time,
                                                    const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::LIGHT_TYPE_INTERNAL_ERRROR;
}

inline TracerError CPULightGroupConstant::ConstructEndpoints(const GPUTransformI** dGlobalTransformArray,
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
                       KCConstructGPULightConstant,
                       //
                       const_cast<GPULightConstant*>(dGPULights),
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

static_assert(IsTracerClass<CPULightGroupConstant>::value,
              "CPULightGroupConstant is not a tracer class");