#pragma once

#include "GPULightP.cuh"
#include "GPUTransformI.h"
#include "DeviceMemory.h"
#include "TypeTraits.h"
#include "MangledNames.h"

class GPULightDirectional final : public GPULightP
{
    private:
        Vector3f                direction;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightDirectional(// Per Light Data
                                                    const Vector3f& direction,
                                                    // Base Class Related
                                                    const TextureRefI<2, Vector3f>& gRad,
                                                    uint16_t mediumId, HitKey,
                                                    const GPUTransformI& gTrans);
                                ~GPULightDirectional() = default;
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

class CPULightGroupDirectional final : public CPULightGroupP<GPULightDirectional>
{
    public:
        TYPENAME_DEF(LightGroup, "Directional");

        static constexpr const char*    DIRECTION_NAME = "direction";

        using Base = CPULightGroupP<GPULightDirectional>;

    private:
        std::vector<Vector3f>           hDirections;

    protected:
    public:
        // Cosntructors & Destructor
                                    CPULightGroupDirectional(const GPUPrimitiveGroupI*,
                                                             const CudaGPU&);
                                    ~CPULightGroupDirectional() = default;

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
        size_t					    UsedCPUMemory() const override;
};

__device__
inline GPULightDirectional::GPULightDirectional(// Per Light Data
                                                const Vector3f& direction,
                                                // Base Class Related
                                                const TextureRefI<2, Vector3f>& gRad,
                                                uint16_t mediumId, HitKey hk,
                                                const GPUTransformI& gTrans)
    : GPULightP(gRad, mediumIndex, hk, gTrans)
    , direction(gTransform.LocalToWorld(direction, true))
{}

__device__
inline void GPULightDirectional::Sample(// Output
                                        float& distance,
                                        Vector3& dir,
                                        float& pdf,
                                        // Input
                                        const Vector3& worldLoc,
                                        // I-O
                                        RandomGPU&) const
{
    dir = -direction;
    distance = FLT_MAX;
    pdf = 1.0f;
}

__device__
inline void GPULightDirectional::GenerateRay(// Output
                                             RayReg&,
                                             // Input
                                             const Vector2i& sampleId,
                                             const Vector2i& sampleMax,
                                             // I-O
                                             RandomGPU& rng,
                                             // Options
                                             bool antiAliasOn) const
{
    // TODO: implement
}

__device__
inline float GPULightDirectional::Pdf(const Vector3& worldDir,
                                      const Vector3& worldPos) const
{
    return 0.0f;
}

__device__
inline float GPULightDirectional::Pdf(float distance,
                                      const Vector3& hitPosition,
                                      const Vector3& direction,
                                      const QuatF& tbnRotation) const
{
    return 0.0f;
}

__device__
inline bool GPULightDirectional::CanBeSampled() const
{
    return false;
}


inline CPULightGroupDirectional::CPULightGroupDirectional(const GPUPrimitiveGroupI* pg,
                                                          const CudaGPU& gpu)
    : Base(*pg, gpu)
{}

inline const char* CPULightGroupDirectional::Type() const
{
    return TypeName();
}

inline size_t CPULightGroupDirectional::UsedCPUMemory() const
{
    size_t totalSize = (Base::UsedCPUMemory() +
                        hDirections.size() * sizeof(Vector3f));

    return totalSize;
}

static_assert(IsTracerClass<CPULightGroupDirectional>::value,
              "CPULightGroupDirectional is not a tracer class");