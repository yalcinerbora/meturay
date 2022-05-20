﻿#pragma once

#include "GPULightP.cuh"
#include "GPUPiecewiseDistribution.cuh"
#include "GPUTransformI.h"
#include "TypeTraits.h"
#include "MangledNames.h"

#include "RayLib/CoordinateConversion.h"

class GPULightSkySphere final : public GPULightP
{
    private:
        const PWCDistributionGPU2D&  distribution;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightSkySphere(// Per Light Data
                                                  const PWCDistributionGPU2D&,
                                                  // Endpoint Related Data
                                                  const TextureRefI<2, Vector3f>& gRad,
                                                  uint16_t mediumIndex, HitKey,
                                                  const GPUTransformI&);
                                ~GPULightSkySphere() = default;
        // Interface
        __device__ void         Sample(// Output
                                       float& distance,
                                       Vector3& direction,
                                       float& pdf,
                                       // Input
                                       const Vector3& worldLoc,
                                       // I-O
                                       RNGeneratorGPUI&) const override;

        __device__ void         GenerateRay(// Output
                                            RayReg&,
                                            // Input
                                            const Vector2i& sampleId,
                                            const Vector2i& sampleMax,
                                            // I-O
                                            RNGeneratorGPUI&,
                                            // Options
                                            bool antiAliasOn = true) const override;
        __device__ float        Pdf(const Vector3& direction,
                                    const Vector3& position) const override;
        __device__ float        Pdf(float distance,
                                    const Vector3& hitPosition,
                                    const Vector3& direction,
                                    const QuatF& tbnRotation) const override;
        __device__ bool         CanBeSampled() const override;

        // Specialize Emit
        __device__ Vector3f     Emit(const Vector3& wo,
                                     const Vector3& pos,
                                     //
                                     const UVSurface&) const override;
};

class CPULightGroupSkySphere final : public CPULightGroupP<GPULightSkySphere>
{
    public:
        TYPENAME_DEF(LightGroup, "SkySphere");

        using Base = CPULightGroupP<GPULightSkySphere>;

    private:
        // CPU Permanent Allocations
        PWCDistributionGroupCPU2D        hLuminanceDistributions;

        DeviceMemory                     gpuDsitributionMem;
        const PWCDistributionGPU2D*      dGPUDistributions;


    protected:
    public:
        // Constructors & Destructor
                                    CPULightGroupSkySphere(const GPUPrimitiveGroupI*,
                                                           const CudaGPU&);
                                    ~CPULightGroupSkySphere() = default;

        const char*				    Type() const override;
		SceneError				    InitializeGroup(const EndpointGroupDataList& lightNodes,
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
        size_t					    UsedGPUMemory() const override;
};

__device__
inline GPULightSkySphere::GPULightSkySphere(// Per Light Data
                                            const PWCDistributionGPU2D& dist,
                                            // Endpoint Related Data
                                            const TextureRefI<2, Vector3f>& gRad,
                                            uint16_t mediumIndex, HitKey hk,
                                            const GPUTransformI& gTransform)
    : GPULightP(gRad, mediumIndex, hk, gTransform)
    , distribution(dist)
{}

__device__
inline void GPULightSkySphere::Sample(// Output
                                      float& distance,
                                      Vector3& dir,
                                      float& pdf,
                                      // Input
                                      const Vector3& worldLoc,
                                      // I-O
                                      RNGeneratorGPUI& rng) const
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
    dir = gTransform.LocalToWorld(dirYUp, true);

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
                                           RNGeneratorGPUI& rng,
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
    Vector3 dirYUp = gTransform.WorldToLocal(direction, true);
    Vector3 dirZUp = Vector3(dirYUp[2], dirYUp[0], dirYUp[1]);
    Vector2 thetaPhi = Utility::CartesianToSphericalUnit(dirZUp);

    // Normalize to generate UV [0, 1]
    // theta range [-pi, pi]
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
    Vector2f index = indexNorm * Vector2f(distribution.Width(), distribution.Height());

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
inline float GPULightSkySphere::Pdf(float distance,
                                    const Vector3& hitPosition,
                                    const Vector3& direction,
                                    const QuatF& tbnRotation) const
{
    return Pdf(direction, Vector3f(0.0f));
}

__device__
inline bool GPULightSkySphere::CanBeSampled() const
{
    return true;
}

__device__
inline Vector3f GPULightSkySphere::Emit(const Vector3& wo,
                                        const Vector3& pos,
                                        //
                                        const UVSurface& surface) const
{
    // Convert Y up from Z up
    Vector3 woTrans = GPUSurface::ToTangent(wo, gTransform.ToLocalRotation());
    Vector3 woZup = -Vector3(woTrans[2], woTrans[0], woTrans[1]);

    // Convert to Spherical Coordinates
    Vector2f thetaPhi = Utility::CartesianToSphericalUnit(woZup);

    // Normalize to generate UV [0, 1]
    // theta range [-pi, pi]
    float u = (thetaPhi[0] + MathConstants::Pi) * 0.5f / MathConstants::Pi;
    // phi range [0, pi]
    float v = 1.0f - (thetaPhi[1] / MathConstants::Pi);

    // Gen Directional vector
    Vector2 uv = Vector2(u, v);
    return gRadianceRef(uv);
}

inline CPULightGroupSkySphere::CPULightGroupSkySphere(const GPUPrimitiveGroupI* pg,
                                                      const CudaGPU& gpu)
    : CPULightGroupP<GPULightSkySphere>(*pg, gpu)
    , dGPUDistributions(nullptr)
{}

inline const char* CPULightGroupSkySphere::Type() const
{
    return TypeName();
}

inline size_t CPULightGroupSkySphere::UsedCPUMemory() const
{
    size_t totalSize = CPULightGroupP<GPULightSkySphere>::UsedCPUMemory();
    return totalSize;
}

inline size_t CPULightGroupSkySphere::UsedGPUMemory() const
{
    size_t totalSize = CPULightGroupP<GPULightSkySphere>::UsedGPUMemory();
    return totalSize;
}

static_assert(IsTracerClass<CPULightGroupSkySphere>::value,
              "CPULightGroupDirectional is not a tracer class");