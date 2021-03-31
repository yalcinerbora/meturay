#pragma once

#include "GPULightI.h"
#include "GPUTransformI.h"
#include "DeviceMemory.h"
#include "TypeTraits.h"
#include "GPUPiecewiseDistribution.cuh"
#include "RayLib/CoordinateConversion.h"

class GPULightSkySphere : public GPULightI
{
    private:
        const GPUDistPiecewiseConst2D&  distribution;
        const GPUTransformI&            transform;
        bool                            isHemi;

    protected:
    public:
        // Constructors & Destructor
        __device__                      GPULightSkySphere(// Per Light Data
                                                          const GPUDistPiecewiseConst2D& dist,
                                                          bool isHemi,
                                                          const GPUTransformI& gTransform,
                                                          // Endpoint Related Data
                                                          HitKey k, uint16_t mediumIndex);
                                        ~GPULightSkySphere() = default;
        // Interface
        __device__ void                 Sample(// Output
                                               float& distance,
                                               Vector3& direction,
                                               float& pdf,
                                               // Input
                                               const Vector3& worldLoc,
                                               // I-O
                                               RandomGPU&) const override;

        __device__ void                 GenerateRay(// Output
                                                    RayReg&,
                                                    // Input
                                                    const Vector2i& sampleId,
                                                    const Vector2i& sampleMax,
                                                    // I-O
                                                    RandomGPU&) const override;

        __device__ PrimitiveId          PrimitiveIndex() const override;
};

class CPULightGroupSkySphere : public CPULightGroupI
{
    public:
        static constexpr const char*    TypeName() { return "SkySphere"; }
        static constexpr const char*    IS_HEMI_NAME = "isHemispherical";

    private:
        DeviceMemory                            memory;
        // CPU Temp Allocation
        std::vector<std::vector<float>>         hLuminances;
        std::vector<Vector2ui>                  hLuminanceSizes;

        std::vector<Byte>                       hIsHemiOptions;
        std::vector<HitKey>                     hHitKeys;
        std::vector<uint16_t>                   hMediumIds;
        std::vector<TransformId>                hTransformIds;
        // CPU Permanent Allocation
        CPUDistGroupPiecewiseConst2D            hLuminanceDistributions;
        // Allocations of the GPU Class
        const GPULightSkySphere*                dGPULights;
        const GPUDistPiecewiseConst2D*          dLuminanceDistributions;
        // GPU pointers to those allocated classes on the CPU
        GPULightList				            gpuLightList;
        uint32_t                                lightCount;

    protected:
    public:
        // Cosntructors & Destructor
                                CPULightGroupSkySphere(const GPUPrimitiveGroupI*);
                                ~CPULightGroupSkySphere() = default;

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
inline GPULightSkySphere::GPULightSkySphere(// Per Light Data
                                            const GPUDistPiecewiseConst2D& dist,
                                            bool isHemi,
                                            const GPUTransformI& gTransform,
                                            // Endpoint Related Data
                                            HitKey k, uint16_t mediumIndex)
    : GPULightI(k, mediumIndex)
    , distribution(dist)
    , transform(gTransform)
    , isHemi(isHemi)
{}

__device__
inline void GPULightSkySphere::Sample(// Output
                                      float& distance,
                                      Vector3& dir,
                                      float& pdf,
                                      // Input
                                      const Vector3& worldLoc,
                                      // I-O
                                      RandomGPU& rng) const
{
    Vector2f uv = distribution.Sample(pdf, rng);

    Vector2f tethaPhi = Vector2f(// [-pi, pi]
                                 (uv[0] * MathConstants::Pi * 2.0f) - MathConstants::Pi,
                                  // [0, pi]
                                 uv[1] * MathConstants::Pi);
    dir = Utility::SphericalToCartesianUnit(tethaPhi);
    // Transform Direction to World Space
    dir = transform.LocalToWorld(dir, true);

    // Sky is very far
    distance = FLT_MAX;   
}

__device__
inline void GPULightSkySphere::GenerateRay(// Output
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
inline PrimitiveId GPULightSkySphere::PrimitiveIndex() const
{
    return INVALID_PRIMITIVE_ID;
}

inline CPULightGroupSkySphere::CPULightGroupSkySphere(const GPUPrimitiveGroupI*)
    : lightCount(0)
    , dGPULights(nullptr)
    , dLuminanceDistributions(nullptr)
{}

inline const char* CPULightGroupSkySphere::Type() const
{
    return TypeName();
}

inline const GPULightList& CPULightGroupSkySphere::GPULights() const
{
    return gpuLightList;
}

inline uint32_t CPULightGroupSkySphere::LightCount() const
{
    return lightCount;
}

inline size_t CPULightGroupSkySphere::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t CPULightGroupSkySphere::UsedCPUMemory() const
{
    size_t totalLumSize = 0;
    for(const auto& lum : hLuminances)
        totalLumSize += lum.size();
    totalLumSize *= sizeof(float);

    size_t totalSize = (hHitKeys.size() * sizeof(HitKey) +
                        hMediumIds.size() * sizeof(uint16_t) +
                        hTransformIds.size() * sizeof(TransformId) +
                        hLuminanceSizes.size() * sizeof(Vector2ui) +
                        totalLumSize);

    return totalSize;
}

static_assert(IsTracerClass<CPULightGroupSkySphere>::value,
              "CPULightGroupDirectional is not a tracer class");