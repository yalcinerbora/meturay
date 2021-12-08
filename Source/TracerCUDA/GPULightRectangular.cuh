#pragma once

#include "GPULightP.cuh"
#include "GPUTransformI.h"
#include "Random.cuh"
#include "TypeTraits.h"
#include "MangledNames.h"

class GPULightRectangular final : public GPULightP
{
    private:
        Vector3                 topLeft;
        Vector3                 right;
        Vector3                 down;
        Vector3                 normal;
        float                   area;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightRectangular(// Per Light Data
                                                    const Vector3& topLeft,
                                                    const Vector3& right,
                                                    const Vector3& down,
                                                    // Endpoint Related Data
                                                    const TextureRefI<2, Vector3f>& gRad,
                                                    uint16_t mediumIndex, HitKey,
                                                    const GPUTransformI& gTransform);
                                ~GPULightRectangular() = default;
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

class CPULightGroupRectangular final : public CPULightGroupP<GPULightRectangular>
{
    public:
        TYPENAME_DEF(LightGroup, "Rectangular");

        static constexpr const char* POSITION_NAME  = "topLeft";
        static constexpr const char* RECT_V0_NAME   = "right";
        static constexpr const char* RECT_V1_NAME   = "down";

        using Base = CPULightGroupP<GPULightRectangular>;

    private:
        std::vector<Vector3f>           hTopLefts;
        std::vector<Vector3f>           hRights;
        std::vector<Vector3f>           hDowns;

    protected:
    public:
        // Constructors & Destructor
                                    CPULightGroupRectangular(const GPUPrimitiveGroupI*,
                                                              const CudaGPU&);
                                    ~CPULightGroupRectangular() = default;

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
inline GPULightRectangular::GPULightRectangular(// Per Light Data
                                                const Vector3& topLeft,
                                                const Vector3& right,
                                                const Vector3& down,
                                                // Endpoint Related Data
                                                const TextureRefI<2, Vector3f>& gRad,
                                                uint16_t mediumIndex, HitKey hk,
                                                const GPUTransformI& gTransform)
    : GPULightP(gRad, mediumIndex, hk, gTransform)
    , topLeft(gTransform.LocalToWorld(topLeft))
    , right(gTransform.LocalToWorld(right, true))
    , down(gTransform.LocalToWorld(down, true))
{
    Vector3 cross = Cross(down, right);
    area = cross.Length();
    normal = cross.Normalize();
}

__device__
inline void GPULightRectangular::Sample(// Output
                                        float& distance,
                                        Vector3& direction,
                                        float& pdf,
                                        // Input
                                        const Vector3& worldLoc,
                                        // I-O
                                        RandomGPU& rng) const
{
    // Sample in the lights local space
    float x = GPUDistribution::Uniform<float>(rng);
    float y = GPUDistribution::Uniform<float>(rng);
    Vector3 position = topLeft + right * x + down * y;

    // Calculate PDF on the local space (it is same on
    direction = position - worldLoc;
    float distanceSqr = direction.LengthSqr();
    distance = sqrt(distanceSqr);
    direction *= (1.0f / distance);

    //float nDotL = max(normal.Dot(-direction), 0.0f);
    float nDotL = abs(normal.Dot(-direction));
    pdf = distanceSqr / (nDotL * area);
    //direction = (position - worldLoc);
    //distance = direction.Length();
    //direction *= (1.0f / distance);

    //// Fake pdf to incorporate square falloff
    //pdf = (distance * distance);
}

__device__ void
inline GPULightRectangular::GenerateRay(// Output
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
inline float GPULightRectangular::Pdf(const Vector3& direction,
                                      const Vector3& worldLoc) const
{
    // Generate 2 Triangle
    Vector3f positions[4] =
    {
        topLeft + right + down,
        topLeft + right,
        topLeft,
        topLeft + down,
    };
    Vector3f normal;
    RayF r(direction, worldLoc);

    Vector3 bCoords;
    float distance;
    bool intersects = r.IntersectsTriangle(bCoords, distance,
                                           positions[0],
                                           positions[1],
                                           positions[2]);
    intersects = r.IntersectsTriangle(bCoords, distance,
                                      positions[0],
                                      positions[2],
                                      positions[3]);

    normal = Cross(down, right).Normalize();

    //float nDotL = max(normal.Dot(-direction), 0.0f);
    float nDotL = abs(normal.Dot(-direction));
    float pdf = (distance * distance) / (nDotL * area);
    return (intersects) ? pdf : 0.0f;
}


__device__
inline float GPULightRectangular::Pdf(float distance,
                                      const Vector3& hitPosition,
                                      const Vector3& direction,
                                      const QuatF& tbnRotation) const
{
    return 1.0f / area;
}

__device__
inline bool GPULightRectangular::CanBeSampled() const
{
    return true;
}

inline CPULightGroupRectangular::CPULightGroupRectangular(const GPUPrimitiveGroupI* pg,
                                                          const CudaGPU& gpu)
    : CPULightGroupP<GPULightRectangular>(*pg, gpu)
{}

inline const char* CPULightGroupRectangular::Type() const
{
    return TypeName();
}

inline size_t CPULightGroupRectangular::UsedCPUMemory() const
{
    size_t totalSize = (CPULightGroupP<GPULightRectangular>::UsedCPUMemory() +
                        hTopLefts.size() * sizeof(Vector3f) +
                        hRights.size() * sizeof(Vector3f) +
                        hDowns.size() * sizeof(Vector3f));

    return totalSize;
}

static_assert(IsTracerClass<CPULightGroupRectangular>::value,
              "CPULightGroupRectangular is not a tracer class");