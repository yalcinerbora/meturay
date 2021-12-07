#pragma once

#include "GPULightP.cuh"
#include "GPUTransformI.h"
#include "Random.cuh"
#include "TypeTraits.h"
#include "MangledNames.h"

class GPULightDisk final : public GPULightP
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
                                             // Base Class Related
                                             const TextureRefI<2, Vector3f>& gRad,
                                             uint16_t mediumId, HitKey,
                                             const GPUTransformI& gTrans);
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

class CPULightGroupDisk final : public CPULightGroupP<GPULightDisk>
{
    public:
        TYPENAME_DEF(LightGroup, "Disk");

        static constexpr const char*    POSITION_NAME = "center";
        static constexpr const char*    NORMAL_NAME = "normal";
        static constexpr const char*    RADIUS_NAME = "radius";

        using Base = CPULightGroupP<GPULightDisk>;

    private:
        //
        std::vector<Vector3f>           hCenters;
        std::vector<Vector3f>           hNormals;
        std::vector<float>              hRadius;

    protected:
    public:
        // Cosntructors & Destructor
                                    CPULightGroupDisk(const GPUPrimitiveGroupI*,
                                                      const CudaGPU&);
                                    ~CPULightGroupDisk() = default;

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
inline GPULightDisk::GPULightDisk(// Per Light Data
                                  const Vector3f& center,
                                  const Vector3f& normal,
                                  const float& radius,
                                   // Base Class Related
                                  const TextureRefI<2, Vector3f>& gRad,
                                  uint16_t mediumId, HitKey hk,
                                  const GPUTransformI& gTrans)
    : GPULightP(gRad, mediumId, hk, gTrans)
    , center(center)
    , normal(normal)
    , radius(radius)
    , area(MathConstants::Pi * radius * radius)
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
    Vector3 localDisk = rotation.ApplyRotation(disk);
    Vector3 localPosition = center + localDisk;
    Vector3 position = gTransform.LocalToWorld(localPosition);

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
                                      RandomGPU&,
                                      // Options
                                      bool antiAliasOn) const
{
    // TODO: Implement
}

__device__
inline float GPULightDisk::Pdf(const Vector3& direction,
                               const Vector3& position) const
{
    RayF r(direction, position);
    r = gTransform.WorldToLocal(r);

    float distance;
    Vector3f planeIntersectPos;
    bool intersects = r.IntersectsPlane(planeIntersectPos, distance,
                                        center, normal);

    intersects &= (planeIntersectPos - center).LengthSqr() <= (radius * radius);
    return (intersects) ? (1.0f / area) : 0.0f;
}

__device__
inline float GPULightDisk::Pdf(float distance,
                               const Vector3& hitPosition,
                               const Vector3& direction,
                               const QuatF& tbnRotation) const
{
    return 1.0f / area;
}

__device__
inline bool GPULightDisk::CanBeSampled() const
{
    return true;
}

inline CPULightGroupDisk::CPULightGroupDisk(const GPUPrimitiveGroupI* pg,
                                            const CudaGPU& gpu)
    : Base(*pg, gpu)
{}

inline const char* CPULightGroupDisk::Type() const
{
    return TypeName();
}

inline size_t CPULightGroupDisk::UsedCPUMemory() const
{
    size_t totalSize = (Base::UsedCPUMemory() +
                        hCenters.size() * sizeof(Vector3f) +
                        hNormals.size() * sizeof(Vector3f) +
                        hRadius.size() * sizeof(float));

    return totalSize;
}

static_assert(IsTracerClass<CPULightGroupDisk>::value,
              "CPULightGroupDisk is not a tracer class");