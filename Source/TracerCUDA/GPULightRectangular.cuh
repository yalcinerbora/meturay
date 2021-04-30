#pragma once

#include "GPULightI.h"
#include "GPUTransformI.h"
#include "DeviceMemory.h"
#include "Random.cuh"
#include "TypeTraits.h"

class GPULightRectangular final : public GPULightI
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
                                                    const GPUTransformI& gTransform,
                                                    // Endpoint Related Data
                                                    HitKey k, uint16_t mediumIndex);
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
                                            RandomGPU&) const override;

        __device__ float        Pdf(const Vector3& direction,
                                    const Vector3 position) const override;

        __device__ bool         CanBeSampled() const override;

        __device__ PrimitiveId  PrimitiveIndex() const override;
};

class CPULightGroupRectangular final : public CPULightGroupI
{
    public:
        static constexpr const char*    TypeName(){return "Rectangular"; }

        static constexpr const char* NAME_POSITION = "topLeft";
        static constexpr const char* NAME_RECT_V0 = "right";
        static constexpr const char* NAME_RECT_V1 = "down";

    private:
        DeviceMemory                    memory;
        //
        std::vector<Vector3f>           hTopLefts;
        std::vector<Vector3f>           hRights;
        std::vector<Vector3f>           hDowns;

        std::vector<HitKey>             hHitKeys;
        std::vector<uint16_t>           hMediumIds;
        std::vector<TransformId>        hTransformIds;
        // Allocations of the GPU Class
        const GPULightRectangular*      dGPULights;
        // GPU pointers to those allocated classes on the CPU
        GPULightList				    gpuLightList;
        uint32_t                        lightCount;

    protected:
    public:
        // Cosntructors & Destructor
                                    CPULightGroupRectangular(const GPUPrimitiveGroupI*);
                                    ~CPULightGroupRectangular() = default;

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
inline GPULightRectangular::GPULightRectangular(// Per Light Data
                                                const Vector3& topLeft,
                                                const Vector3& right,
                                                const Vector3& down,
                                                const GPUTransformI& gTransform,
                                                // Endpoint Related Data
                                                HitKey k, uint16_t mediumIndex)
    : GPULightI(k, mediumIndex)
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

    //// Fake pdf to incorporate square faloff
    //pdf = (distance * distance);
}

__device__ void
inline GPULightRectangular::GenerateRay(// Output
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
inline float GPULightRectangular::Pdf(const Vector3& direction,
                                      const Vector3 worldLoc) const
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
inline bool GPULightRectangular::CanBeSampled() const
{
    return true;
}

__device__
inline PrimitiveId GPULightRectangular::PrimitiveIndex() const
{
    return INVALID_PRIMITIVE_ID;
}

inline CPULightGroupRectangular::CPULightGroupRectangular(const GPUPrimitiveGroupI*)
    : lightCount(0)
    , dGPULights(nullptr)
{}

inline const char* CPULightGroupRectangular::Type() const
{
    return TypeName();
}

inline const GPULightList& CPULightGroupRectangular::GPULights() const
{
    return gpuLightList;
}

inline uint32_t CPULightGroupRectangular::LightCount() const
{
    return lightCount;
}

inline size_t CPULightGroupRectangular::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t CPULightGroupRectangular::UsedCPUMemory() const
{
    size_t totalSize = (hHitKeys.size() * sizeof(HitKey) +
                        hMediumIds.size() * sizeof(uint16_t) +
                        hTransformIds.size() * sizeof(TransformId) +
                        hTopLefts.size() * sizeof(Vector3f) +
                        hRights.size() * sizeof(Vector3f) +
                        hDowns.size() * sizeof(Vector3f));

    return totalSize;
}

static_assert(IsTracerClass<CPULightGroupRectangular>::value,
              "CPULightGroupRectangular is not a tracer class");