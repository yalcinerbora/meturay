#pragma once

#include "GPULightI.h"
#include "GPUTransformI.h"
#include "DeviceMemory.h"
#include "TypeTraits.h"

class GPULightPoint final : public GPULightI
{
    private:
        Vector3f                position;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightPoint(// Per Light Data
                                              const Vector3f& position,
                                              const GPUTransformI& gTransform,
                                              // Endpoint Related Data
                                              HitKey k, uint16_t mediumIndex);
                                ~GPULightPoint() = default;
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

class CPULightGroupPoint final : public CPULightGroupI
{
    public:
        static constexpr const char*    TypeName(){return "Point"; }

        static constexpr const char*    NAME_POSITION = "position";

    private:
        DeviceMemory                    memory;
        //
        std::vector<Vector3f>           hPositions;

        std::vector<HitKey>             hHitKeys;
        std::vector<uint16_t>           hMediumIds;
        std::vector<TransformId>        hTransformIds;
        // Allocations of the GPU Class
        const GPULightPoint*            dGPULights;
        // GPU pointers to those allocated classes on the CPU
        GPULightList				    gpuLightList;
        uint32_t                        lightCount;

    protected:
    public:
        // Cosntructors & Destructor
                                    CPULightGroupPoint(const GPUPrimitiveGroupI*);
                                    ~CPULightGroupPoint() = default;

        const char*				    Type() const override;
		const GPULightList&		    GPULights() const override;
		SceneError				    InitializeGroup(const ConstructionDataList& lightNodes,
                                                    const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                    const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                    const MaterialKeyListing& allMaterialKeys,
								    				double time,
								    				const std::string& scenePath) override;
		SceneError				    ChangeTime(const NodeListing& lightNodes, double time,
								    		   const std::string& scenePath) override;
		TracerError				    ConstructLights(const CudaSystem&,
                                                    const GPUTransformI**) override;
		uint32_t				    LightCount() const override;

        // Luminance Dist Related
		bool						RequiresLuminance() const override;
		const std::vector<HitKey>&	AcquireMaterialKeys() const override;
		TracerError					GenerateLumDistribution(const std::vector<std::vector<float>>& luminance,
															const std::vector<Vector2ui>& dimension,
															const CudaSystem&) override;

		size_t					    UsedGPUMemory() const override;
		size_t					    UsedCPUMemory() const override;
};

__device__
inline GPULightPoint::GPULightPoint(// Per Light Data
                                    const Vector3f& position,
                                    const GPUTransformI& gTransform,
                                    // Endpoint Related Data
                                    HitKey k, uint16_t mediumIndex)
    : GPULightI(k, mediumIndex)
    , position(gTransform.LocalToWorld(position))
{}

__device__
inline void GPULightPoint::Sample(// Output
                                  float& distance,
                                  Vector3& direction,
                                  float& pdf,
                                  // Input
                                  const Vector3& worldLoc,
                                  // I-O
                                  RandomGPU&) const
{
    direction = (position - worldLoc);
    distance = direction.Length();
    direction *= (1.0f / distance);

    // Fake pdf to incorporate square faloff
    pdf = (distance * distance);
}

__device__
inline void GPULightPoint::GenerateRay(// Output
                                       RayReg&,
                                       // Input
                                       const Vector2i& sampleId,
                                       const Vector2i& sampleMax,
                                       // I-O
                                       RandomGPU&) const
{
    // TODO: Implement
}

__device__
inline PrimitiveId GPULightPoint::PrimitiveIndex() const
{
    return INVALID_PRIMITIVE_ID;
}

inline CPULightGroupPoint::CPULightGroupPoint(const GPUPrimitiveGroupI*)
    : lightCount(0)
    , dGPULights(nullptr)
{}

inline const char* CPULightGroupPoint::Type() const
{
    return TypeName();
}

inline const GPULightList& CPULightGroupPoint::GPULights() const
{
    return gpuLightList;
}

inline uint32_t CPULightGroupPoint::LightCount() const
{
    return lightCount;
}

inline size_t CPULightGroupPoint::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t CPULightGroupPoint::UsedCPUMemory() const
{
    size_t totalSize = (hHitKeys.size() * sizeof(HitKey) +
                        hMediumIds.size() * sizeof(uint16_t) +
                        hTransformIds.size() * sizeof(TransformId) +
                        hPositions.size() * sizeof(Vector3f));

    return totalSize;
}

inline bool CPULightGroupPoint::RequiresLuminance() const
{
    return false;
}

inline const std::vector<HitKey>& CPULightGroupPoint::AcquireMaterialKeys() const
{
    return hHitKeys;
}

inline TracerError CPULightGroupPoint::GenerateLumDistribution(const std::vector<std::vector<float>>& luminance,
                                                               const std::vector<Vector2ui>& dimension,
                                                               const CudaSystem&)
{
    return TracerError::LIGHT_GROUP_CAN_NOT_GENERATE_DISTRIBUTION;
}

static_assert(IsTracerClass<CPULightGroupPoint>::value,
              "CPULightGroupPoint is not a tracer class");