#pragma once

#include "GPULightI.h"
#include "TypeTraits.h"
#include "MangledNames.h"

class GPULightNull : public GPULightI
{
    public:
    // Interface
    __device__ Vector3f     Emit(const Vector3& wo,
                                 const Vector3& pos,
                                 //
                                 const UVSurface&) const  override { return Zero3f;}
    __device__ uint32_t     GlobalLightIndex() const  override { return UINT32_MAX; }
    __device__ void         SetGlobalLightIndex(uint32_t) override {}
    __device__ bool         IsPrimitiveBackedLight() const  override { return false; }
    __device__ void         Sample(// Output
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& position,
                                   // I-O
                                   RandomGPU&) const  override {}
    // Generate a Ray from this endpoint
    __device__ void         GenerateRay(// Output
                                        RayReg&,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RandomGPU&,
                                        // Options
                                        bool antiAliasOn = true) const override {}
    __device__ float        Pdf(const Vector3& direction,
                                const Vector3& position) const override { return 0.0f; }
    __device__ float        Pdf(float distance,
                                const Vector3& hitPosition,
                                const Vector3& direction,
                                const QuatF& tbnRotation) const override { return 0.0f; }
    __device__ bool         CanBeSampled() const override { return false; }

};

class CPULightGroupNull final : public CPULightGroupI
{
    public:
        TYPENAME_DEF(LightGroup, "Null");

        // Only use uv surface for now
        using Surface = UVSurface;
        // GPU Work Class will use this to specify the templated kernel
        using PrimitiveGroup        = GPUPrimitiveEmpty;
        using GPUType               = GPULightNull;
        using HitData               = EmptyHit;
        using PrimitiveData         = EmptyData;
        using SF                    = SurfaceFunc<Surface, EmptyHit, EmptyData>;
        static constexpr SF         SurfF = DefaultGenUvSurface<EmptyHit, EmptyData>;

    private:
        std::vector<HitKey> hkList;
        GPUEndpointList     epList;
        GPULightList        lList;
        const CudaGPU&      gpu;

    protected:
    public:
        // Constructors & Destructor
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
        const GPULightList&         GPULights() const override { return lList; }
        const GPUEndpointList&      GPUEndpoints() const override {return epList;}
        const CudaGPU&              GPU() const override { return gpu; }
        uint32_t				        EndpointCount() const override { return 0; }
        const std::vector<HitKey>&  PackedHitKeys() const override { return hkList; }
        uint32_t                    MaxInnerId() const override { return 1; }

        size_t						UsedGPUMemory() const override { return 0; }
        size_t						UsedCPUMemory() const override { return 0; }

        const GPUType*              GPULightsDerived() const {return nullptr;}
};

inline CPULightGroupNull::CPULightGroupNull(const GPUPrimitiveGroupI*,
                                            const CudaGPU& gpu)
    : gpu(gpu)
{}

inline const char* CPULightGroupNull::Type() const
{
    return TypeName();
}

inline SceneError CPULightGroupNull::InitializeGroup(const EndpointGroupDataList&,
                                                     const TextureNodeMap&,
                                                     const std::map<uint32_t, uint32_t>&,
                                                     const std::map<uint32_t, uint32_t>&,
                                                     uint32_t batchId, double,
                                                     const std::string&)
{
    hkList.push_back(HitKey::CombinedKey(batchId, 0));
    return SceneError::OK;
}

inline SceneError CPULightGroupNull::ChangeTime(const NodeListing&, double,
                                                const std::string&)
{
    return SceneError::OK;
}

inline TracerError CPULightGroupNull::ConstructEndpoints(const GPUTransformI**,
                                                         const CudaSystem&)
{
    return TracerError::OK;
}

static_assert(IsTracerClass<CPULightGroupNull>::value,
              "CPULightGroupNull is not a tracer class");