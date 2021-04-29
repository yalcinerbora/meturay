#pragma once

#include "GPULightI.h"
#include "GPUTransformI.h"
#include "DeviceMemory.h"
#include "Random.cuh"
#include "TypeTraits.h"

class GPULightDisk final : public GPULightI
{
    private:
        Vector3f                center;
        Vector3f                normal;
        float                   radius;
        float                   area;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightDisk(// Per Light Data
                                             const Vector3f& center,
                                             const Vector3f& normal,
                                             const float& radius,
                                             const GPUTransformI& gTransform,
                                             // Endpoint Related Data
                                             HitKey k, uint16_t mediumIndex);
                                ~GPULightDisk() = default;
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
        __device__ float        Pdf(const Vector3& direction,
                                    const Vector3 position) const override;

        __device__ bool         CanBeSampled() const override;

        __device__ PrimitiveId  PrimitiveIndex() const override;
};

class CPULightGroupDisk final : public CPULightGroupI
{
    public:
        static constexpr const char*    TypeName(){return "Disk"; }

        static constexpr const char*    NAME_POSITION = "center";
        static constexpr const char*    NAME_NORMAL = "normal";
        static constexpr const char*    NAME_RADIUS = "radius";

    private:
        DeviceMemory                    memory;
        //
        std::vector<Vector3f>           hCenters;
        std::vector<Vector3f>           hNormals;
        std::vector<float>              hRadius;

        std::vector<HitKey>             hHitKeys;
        std::vector<uint16_t>           hMediumIds;
        std::vector<TransformId>        hTransformIds;
        // Allocations of the GPU Class
        const GPULightDisk*             dGPULights;
        // GPU pointers to those allocated classes on the CPU
        GPULightList				    gpuLightList;
        uint32_t                        lightCount;

    protected:
    public:
        // Cosntructors & Destructor
                                    CPULightGroupDisk(const GPUPrimitiveGroupI*);
                                    ~CPULightGroupDisk() = default;

        const char*				    Type() const override;
		const GPULightList&		    GPULights() const override;
		SceneError				    InitializeGroup(const LightGroupDataList& lightNodes,
                                                    const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                    const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                    const MaterialKeyListing& allMaterialKeys,
								    				double time,
								    				const std::string& scenePath) override;
		SceneError				    ChangeTime(const NodeListing& lightNodes, double time,
								    		   const std::string& scenePath) override;
		TracerError				    ConstructLights(const CudaSystem&,
                                                    const GPUTransformI**,
                                                    const KeyMaterialMap&) override;
		uint32_t				    LightCount() const override;
        
		size_t					    UsedGPUMemory() const override;
		size_t					    UsedCPUMemory() const override;
};

__device__
inline GPULightDisk::GPULightDisk(// Per Light Data
                                  const Vector3f& center,
                                  const Vector3f& normal,
                                  const float& radius,
                                  const GPUTransformI& gTransform,
                                  // Endpoint Related Data
                                  HitKey k, uint16_t mediumIndex)
    : GPULightI(k, mediumIndex)
    , center(gTransform.LocalToWorld(center))
    , normal(gTransform.LocalToWorld(normal, true))
    , radius(radius)
    , area(MathConstants::Pi* radius* radius)
{
    // TODO: check the scale on the transform
    // since this wont work if transform contains scale
}

__device__
inline void GPULightDisk::Sample(// Output
                                 float& distance,
                                 Vector3& direction,
                                 float& pdf,
                                 // Input
                                 const Vector3& worldLoc,
                                 // I-O
                                 RandomGPU& rng) const
{
    float r = GPUDistribution::Uniform<float>(rng) * radius;
    float tetha = GPUDistribution::Uniform<float>(rng) * 2.0f * MathConstants::Pi;

    // Aligned to Axis Z
    Vector3 disk = Vector3(sqrt(r) * cos(tetha),
                           sqrt(r) * sin(tetha),
                           0.0f);

    // Rotate to disk normal
    QuatF rotation = Quat::RotationBetweenZAxis(normal);
    Vector3 worldDisk = rotation.ApplyRotation(disk);
    Vector3 position = center + worldDisk;

    direction = position - worldLoc;
    float distanceSqr = direction.LengthSqr();
    distance = sqrt(distanceSqr);
    direction *= (1.0f / distance);

    //float nDotL = max(normal.Dot(-direction), 0.0f);
    float nDotL = abs(normal.Dot(-direction));
    pdf = distanceSqr / (nDotL * area);
}

__device__
inline void GPULightDisk::GenerateRay(// Output
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
inline float GPULightDisk::Pdf(const Vector3& direction,
                                const Vector3 position) const
{
    return ...;
}

__device__
inline bool GPULightDisk::CanBeSampled() const
{
    return true;
}

__device__
inline PrimitiveId GPULightDisk::PrimitiveIndex() const
{
    return INVALID_PRIMITIVE_ID;
}

inline CPULightGroupDisk::CPULightGroupDisk(const GPUPrimitiveGroupI*)
    : lightCount(0)
    , dGPULights(nullptr)
{}

inline const char* CPULightGroupDisk::Type() const
{
    return TypeName();
}

inline const GPULightList& CPULightGroupDisk::GPULights() const
{
    return gpuLightList;
}

inline uint32_t CPULightGroupDisk::LightCount() const
{
    return lightCount;
}

inline size_t CPULightGroupDisk::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t CPULightGroupDisk::UsedCPUMemory() const
{
    size_t totalSize = (hHitKeys.size() * sizeof(HitKey) +
                        hMediumIds.size() * sizeof(uint16_t) +
                        hTransformIds.size() * sizeof(TransformId) +
                        hCenters.size() * sizeof(Vector3f) +
                        hNormals.size() * sizeof(Vector3f) +
                        hRadius.size() * sizeof(float));

    return totalSize;
}

static_assert(IsTracerClass<CPULightGroupDisk>::value,
              "CPULightGroupDisk is not a tracer class");