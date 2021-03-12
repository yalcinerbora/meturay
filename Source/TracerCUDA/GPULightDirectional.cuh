#pragma once

#include "GPULightI.h"
#include "GPUTransformI.h"
#include "DeviceMemory.h"
#include "TypeTraits.h"

class GPULightDirectional : public GPULightI
{
    private:        
        Vector3f                direction;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightDirectional(// Per Light Data
                                                    const Vector3f& direction,
                                                    const GPUTransformI& gTransform,
                                                    // Endpoint Related Data
                                                    HitKey k, uint16_t mediumIndex);
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
                                            RandomGPU&) const override;

        __device__ PrimitiveId  PrimitiveIndex() const override;
};


class CPULightGroupDirectional : public CPULightGroupI
{
    public:
        static constexpr const char*    TypeName() { return "Directional"; }

        static constexpr const char*    NAME_DIRECTION = "direction";

    private:
        DeviceMemory                    memory;
        //
        std::vector<Vector3f>           hDirections;

        std::vector<HitKey>             hHitKeys;
        std::vector<uint16_t>           hMediumIds;
        std::vector<TransformId>        hTransformIds;
        // Allocations of the GPU Class
        const GPULightDirectional*      dGPULights;
        // GPU pointers to those allocated classes on the CPU
        GPULightList				    gpuLightList;
        uint32_t                        lightCount;
        
    protected:
    public:
        // Cosntructors & Destructor
                                CPULightGroupDirectional(const GPUPrimitiveGroupI*);
                                ~CPULightGroupDirectional() = default;


        const char*				Type() const override;
		const GPULightList&		GPULights() const override;
		SceneError				InitializeGroup(const ConstructionDataList& lightNodes,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                const MaterialKeyListing& allMaterialKeys,
												double time,
												const std::string& scenePath) override;
		SceneError				ChangeTime(const NodeListing& lightNodes, double time,
										   const std::string& scenePath) override;
		TracerError				ConstructLights(const CudaSystem&,
                                                const GPUTransformI**) override;
		uint32_t				LightCount() const override;

		size_t					UsedGPUMemory() const override;
        size_t					UsedCPUMemory() const override; 
};

__device__
inline GPULightDirectional::GPULightDirectional(// Per Light Data
                                                const Vector3f& direction,
                                                const GPUTransformI& gTransform,
                                                // Endpoint Related Data
                                                HitKey k, uint16_t mediumIndex)
    : GPUEndpointI(k, mediumIndex)
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
                                             RandomGPU& rng) const
{
    // TODO: implement
}

__device__ 
inline PrimitiveId GPULightDirectional::PrimitiveIndex() const
{
    return INVALID_PRIMITIVE_ID;
}

inline CPULightGroupDirectional::CPULightGroupDirectional(const GPUPrimitiveGroupI*)
    : CPULightGroupI()
    , lightCount(0)
    , dGPULights(nullptr)
{}

inline const char* CPULightGroupDirectional::Type() const
{
    return TypeName();
}

inline const GPULightList& CPULightGroupDirectional::GPULights() const
{
    return gpuLightList;
}

inline uint32_t CPULightGroupDirectional::LightCount() const
{
    return lightCount;
}

inline size_t CPULightGroupDirectional::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t CPULightGroupDirectional::UsedCPUMemory() const
{
    size_t totalSize = (hHitKeys.size() * sizeof(HitKey) +
                        hMediumIds.size() * sizeof(uint16_t) +
                        hTransformIds.size() * sizeof(TransformId) + 
                        hDirections.size() * sizeof(Vector3f));

    return totalSize;
}

static_assert(IsTracerClass<CPULightGroupDirectional>::value,
              "CPULightGroupDirectional is not a tracer class");