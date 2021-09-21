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
                                                    RandomGPU&,
                                                    // Options
                                                    bool antiAliasOn = true) const override;

        __device__ float                Pdf(const Vector3& direction,
                                            const Vector3& position) const override;

        __device__ bool                 CanBeSampled() const override;

        __device__ PrimitiveId          PrimitiveIndex() const override;
};

class CPULightGroupSkySphere : public CPULightGroupI
{
    public:
        static constexpr const char*    TypeName() { return "SkySphere"; }
        static constexpr const char*    IS_HEMI_NAME = "isHemispherical";

    private:
        DeviceMemory                            memory;
        // CPU Temp Allocations
        std::vector<Byte>                       hIsHemiOptions;
        std::vector<HitKey>                     hHitKeys;
        std::vector<uint16_t>                   hMediumIds;
        std::vector<TransformId>                hTransformIds;
        // CPU Permanent Allocations
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
    Vector2f index;
    Vector2f uv = distribution.Sample(pdf, index, rng);
    Vector2f thetaPhi = Vector2f(// [-pi, pi]
                                 (uv[0] * MathConstants::Pi * 2.0f) - MathConstants::Pi,
                                  // [0, pi]
                                 (1.0f - uv[1]) * MathConstants::Pi);
    Vector3 dirZUp = Utility::SphericalToCartesianUnit(thetaPhi);
    // Spherical Coords calculates as Z up change it to Y up
    Vector3 dirYUp = Vector3(dirZUp[1], dirZUp[2], dirZUp[0]);
    // Transform Direction to World Space
    dir = transform.LocalToWorld(dirYUp, true);

    // Convert to solid angle pdf
    // http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources.html
    float sinPhi = sin(thetaPhi[1]);
    if(sinPhi == 0.0f) pdf = 0.0f;
    else pdf = pdf / (2.0f * MathConstants::Pi * MathConstants::Pi * sinPhi);

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
                                           RandomGPU& rng,
                                           // Options
                                           bool antiAliasOn) const
{
    // TODO: implement
}

__device__
inline float GPULightSkySphere::Pdf(const Vector3& direction,
                                    const Vector3& position) const
{
    // Convert to spherical coordinates
    Vector3 dirYUp = transform.WorldToLocal(direction, true);
    Vector3 dirZup = -Vector3(dirYUp[2], dirYUp[0], dirYUp[1]);
    Vector2 thetaPhi = Utility::CartesianToSphericalUnit(dirZup);
        
    // Normalize to generate UV [0, 1]
    // tetha range [-pi, pi]
    float u = (thetaPhi[0] + MathConstants::Pi) * 0.5f / MathConstants::Pi;
    // If we are at edge point (u == 1) make it zero since 
    // piecewise constant function will not have that pdf (out of bounds)
    u = (u == 1.0f) ? 0.0f : u;
    // phi range [0, pi]
    float v = 1.0f - (thetaPhi[1] / MathConstants::Pi);
    // If (v == 1) then again pdf of would be out of bounds.
    // make it inbound
    v = (v == 1.0f) ? (v - MathConstants::SmallEpsilon) : v;

    // If we are at the edge point
    Vector2f indexNorm = Vector2f(u, v);

    // Expand to size
    Vector2f index = indexNorm * (distribution.Width(), distribution.Height());

    // Fetch Conditional/Marginal Probs
    float pdf = distribution.Pdf(index);

    // Convert to solid angle pdf
    // http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources.html
    float sinPhi = sin(thetaPhi[1]);
    if(sinPhi == 0.0f) pdf = 0.0f;
    else pdf = pdf / (2.0f * MathConstants::Pi * MathConstants::Pi * sinPhi);

    return pdf;
}

__device__
inline bool GPULightSkySphere::CanBeSampled() const
{
    return true;
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
    size_t totalSize = (memory.Size() +
                        hLuminanceDistributions.UsedGPUMemory());
    return totalSize;
}

inline size_t CPULightGroupSkySphere::UsedCPUMemory() const
{    
    size_t totalSize = (hHitKeys.size() * sizeof(HitKey) +
                        hMediumIds.size() * sizeof(uint16_t) +
                        hTransformIds.size() * sizeof(TransformId));
    return totalSize;
}

static_assert(IsTracerClass<CPULightGroupSkySphere>::value,
              "CPULightGroupDirectional is not a tracer class");